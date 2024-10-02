import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import os
# from torchprofile import profile_macs

"""
PNLQ Config Expected Structure:
{
    "llaf_quantization": [
        {
            "input_quantization_type" : str,
            "max_input_value": float,
            "threshold": float,
            "single_value_prediction": float [optional],
            "n_bits_act": int,
            "n_bits_wgt": int [optional]
        }
    ],
    "awsm_quantization": [
        {
            "attn_score_threshold": float,
            "sum_exp_threshold": float,
            "combine_threshold": str,
            "max_input_value": float,
            "n_bits_act": int,
        }
    ],
    "profiling": str [optional],
}
"""

class QuantizedLLAF(nn.Module):
    """
    Quantized Linear Layer with Activation Function
    """
    __constants__ = ['approximate', 'in_features', 'out_features']
    approxmate: str
    in_features: int
    out_features: int
    weight: Tensor

    llaf_index = 0

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none', act_fn = None, config = None) -> None:
        super(QuantizedLLAF, self).__init__()
        self.index = QuantizedLLAF.llaf_index
        self.approximate = approximate
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.act_fn = act_fn
        assert self.act_fn is not None, "Activation function must be provided."

        self.config = config
        self.load_config(config, QuantizedLLAF.llaf_index)
        QuantizedLLAF.llaf_index += 1
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def load_config(self, config, llaf_index) -> None:
        """
        Load configuration for quantization and logging.

        """

        if config is None:
            self.quantization = False
            return
        if 'llaf_quantization' in config:
            self.quantization = True
            if len(config['llaf_quantization']) == 1:
                q_config = config['llaf_quantization'][0]
            elif len(config['llaf_quantization']) > llaf_index:
                q_config = config['llaf_quantization'][llaf_index]
            else:
                print(f"LLAF Quantization configuration for layer {llaf_index} not found. Using configuration at index 0.")
                q_config = config['llaf_quantization'][0]

            self.input_quantization_type = q_config['input_quantization_type']
            self.max_input_value = torch.tensor(q_config['max_input_value']).cuda()
            self.threshold = q_config['threshold']
            self.n_bits_act = q_config['n_bits_act']
            # quantizing weights and single value prediction are optional, all others are required
            self.n_bits_wgt = q_config['n_bits_wgt'] if 'n_bits_wgt' in q_config else None
            self.single_value_prediction = q_config['single_value_prediction'] if 'single_value_prediction' in q_config else None

        else:
            self.quantization = False

    def forward(self, input: Tensor) -> Tensor:
        collect_frequency_stats(input, "llaf_input", self.config)
        collect_frequency_stats(self.weight, "llaf_weight", self.config)
        collect_frequency_stats(self.bias, "llaf_bias", self.config)
        collect_frequency_stats(input, f"llaf_input_{self.index}", self.config)
        collect_frequency_stats(self.weight, f"llaf_weight_{self.index}", self.config)
        collect_frequency_stats(self.bias, f"llaf_bias_{self.index}", self.config)

        full_precision_preactivation = input @ self.weight.t() + self.bias

        collect_frequency_stats(full_precision_preactivation, "llaf_full_precision_preactivation", self.config)
        collect_frequency_stats(full_precision_preactivation, f"llaf_full_precision_preactivation_{self.index}", self.config)

        # full precision computation
        full_precision_output = self.act_fn(full_precision_preactivation)

        collect_frequency_stats(full_precision_output, "llaf_full_precision_output", self.config)
        collect_frequency_stats(full_precision_output, f"llaf_full_precision_output_{self.index}", self.config)

        if self.quantization:
            max_weight_value = torch.max(torch.abs(self.weight),dim=-1, keepdim=True).values

            quantized_input = quantize(input, self.n_bits_act, self.max_input_value)
            if self.n_bits_wgt is not None:
                quantized_weights = quantize(self.weight, self.n_bits_wgt, max_weight_value)
            else:
                quantized_weights = self.weight
            quantized_bias = quantize(self.bias, self.n_bits_act, self.max_input_value)

            if self.n_bits_wgt is not None:
                scaled_quantized_prod = (quantized_input @ quantized_weights.t())/(self.weight.shape[-1])/2**((self.n_bits_act + self.n_bits_wgt)/2)
            else:
                scaled_quantized_prod = (quantized_input @ quantized_weights.t())/(self.weight.shape[-1])/2**((self.n_bits_act))


            quantized_input = quantize(input, self.n_bits_act, self.max_input_value)
            quantized_bias = quantize(self.bias, self.n_bits_act, self.max_input_value)
            if self.n_bits_wgt is not None:
                quantized_weights = quantize(self.weight, self.n_bits_wgt, max_weight_value)

            quantized_output = self.act_fn((quantized_input @ quantized_weights.t()) + quantized_bias)

            below_threshold_mask = torch.where(scaled_quantized_prod < self.threshold, 1, 0)

            actual_sign_value = torch.where(full_precision_output <= 0, 1, 0)
            collect_prediction_stats(below_threshold_mask, actual_sign_value)
            output = torch.where(below_threshold_mask == 1, quantized_output, full_precision_output)

            collect_frequency_stats(output, "llaf_quantized_output", self.config)
            collect_frequency_stats(output, f"llaf_quantized_output_{self.index}", self.config)
            collect_quantization_error_stats(full_precision_output, output, f"llaf_output", config = self.config)
            collect_quantization_error_stats(full_precision_output, output, f"llaf_output_{self.index}", config = self.config)
            
            return output
        else:
            return full_precision_output

class QuantizedAWSM(nn.Module):
    """
    Softmax function that quantizes where error is mitigated
    """
    awsm_index = 0

    def __init__(self, dim, config = None) -> None:
        super(QuantizedAWSM, self).__init__()
        self.softmax = nn.Softmax(dim)
        self.dim = dim
        self.index = QuantizedAWSM.awsm_index

        if config is not None and 'awsm_quantization' in config:
            self.quantization = True
            if len(config['awsm_quantization']) == 1:
                q_config = config['awsm_quantization'][0]
            elif len(config['awsm_quantization']) > QuantizedAWSM.awsm_index:
                q_config = config['awsm_quantization'][QuantizedAWSM.awsm_index]
            else:
                print(f"AWSM Quantization configuration for layer {QuantizedAWSM.awsm_index} not found. Using configuration at index 0.")
                q_config = config['awsm_quantization'][0]

            self.attn_score_threshold = q_config['attn_score_threshold']
            self.sum_exp_threshold = q_config['sum_exp_threshold']
            self.max_input_value = torch.tensor(q_config['max_input_value']).cuda()
            self.n_bits_act = q_config['n_bits_act']
            self.combine_threshold = q_config['combine_threshold']
        else:
            self.quantization = False
        self.config = config

        QuantizedAWSM.awsm_index += 1

    def quantize_awsm_input(self, input):
        if self.quantization:
            input = quantize(input, self.n_bits_act, self.max_input_value)
        return input
    
    
    def forward(self, input: Tensor, quantized_input: Tensor) -> Tensor:
        collect_frequency_stats(input, "awsm_input", self.config)
        collect_frequency_stats(input, f"awsm_input_{self.index}", self.config)


        if self.quantization:
            collect_frequency_stats(quantized_input, "awsm_quantized_input", self.config)
            collect_frequency_stats(quantized_input, f"awsm_quantized_input_{self.index}", self.config)
            quantized_sum_exp = torch.sum(torch.exp(quantized_input), self.dim, keepdim=True)
            attn_score_threshold_mask = torch.where(quantized_input <= self.attn_score_threshold, 1, 0)
            sum_exp_threshold_mask = torch.where(quantized_sum_exp >= self.sum_exp_threshold, 1, 0)
            if self.combine_threshold == "and":
                quantized_mask = torch.minimum(attn_score_threshold_mask, sum_exp_threshold_mask)
            elif self.combine_threshold == "or":
                quantized_mask = torch.maximum(attn_score_threshold_mask, sum_exp_threshold_mask)

            collect_awsm_stats(attn_score_threshold_mask, sum_exp_threshold_mask, quantized_mask)

            pnlq_output = torch.where(quantized_mask == 1, quantized_input, input)
            collect_quantization_error_stats(input, pnlq_output, f"awsm_output", dim = self.dim, config = self.config)
            collect_quantization_error_stats(input, pnlq_output, f"awsm_output_{self.index}", dim = self.dim, config = self.config)
        else:
            pnlq_output = input

        return F.softmax(pnlq_output, dim = self.dim)

def collect_prediction_stats(predicted, actual):
    precision = torch.sum(torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 1, 1, 0))) / torch.sum(torch.where(predicted == 1, 1, 0))
    recall = torch.sum(torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 1, 1, 0))) / torch.sum(torch.where(actual == 1, 1, 0))
    negative_predictive_value = torch.sum(torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 0, 1, 0))) / torch.sum(torch.where(predicted == 0, 1, 0))
    true_negative_rate = torch.sum(torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 0, 1, 0))) / torch.sum(torch.where(actual == 0, 1, 0))
    false_positive_rate = torch.sum(torch.mul(torch.where(predicted != actual, 1, 0), torch.where(predicted == 1, 1, 0))) / torch.sum(torch.where(actual == 0, 1, 0))
    false_negative_rate = torch.sum(torch.mul(torch.where(predicted != actual, 1, 0), torch.where(predicted == 0, 1, 0))) / torch.sum(torch.where(actual == 1, 1, 0))

    total_values = actual.numel()
    proportion_pred_neg = torch.sum(torch.where(predicted == 1, 1, 0)) / total_values

    with open("prediction_stats.csv", 'a') as results_file:
        results_file.write(f"{precision},{recall},{negative_predictive_value},{true_negative_rate},{false_positive_rate},{false_negative_rate},{proportion_pred_neg},{total_values}\n")

def collect_awsm_stats(attn_score_threshold_mask, sum_exp_threshold_mask, quantized_mask):
    prop_quantized_attn_score = torch.sum(attn_score_threshold_mask) / attn_score_threshold_mask.numel()
    prop_quantized_sum_exp = torch.sum(sum_exp_threshold_mask) / sum_exp_threshold_mask.numel()
    prop_quantized = torch.sum(quantized_mask) / quantized_mask.numel()
    total_values = quantized_mask.numel()
    with open("awsm_stats.csv", 'a') as results_file:
        results_file.write(f"{prop_quantized_attn_score},{prop_quantized_sum_exp},{prop_quantized},{total_values}\n")

@torch.no_grad()
def quantize(input, n_bits, scales = None, quantization_type = "fixed"):
    # n_bits: number of non-sign bits
    # scales: scale for each channel, typically max abs value of the input
    # adapted from https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/fake_quant.py

    if quantization_type == "per_channel":
        scales = input.abs().max(dim=-1, keepdim=True)[0]
    elif quantization_type == "per_tensor":
        scales = input.abs().max()
    elif quantization_type == "per_token":
        input_shape = input.shape
        input.view(-1, input_shape[-1])
        scales = input.abs().max(dim=-1, keepdim=True)[0]

    q_max = 2 ** (n_bits - 1) - 1
    scales = scales.clamp(min=1e-5).div(q_max)
    # clip to [-q_max - 1, q_max] when quantizing
    return torch.round(input / scales).clamp(min=-q_max - 1, max=q_max) * scales

def dequantize_fixed_offset(input, n_bits, scales):
    num_values_in_quantization = 2**n_bits
    return torch.sign(input) * (torch.abs(input) - 1) / num_values_in_quantization * scales

def process_prediction_stats(model_str, metric_str, write_newline = True):
    if os.path.exists("prediction_stats.csv"):
        with open("prediction_stats.csv", 'r') as stats_file:
            lines = stats_file.readlines()
            total_values = 0
            precision_sum = 0.0
            recall_sum = 0.0
            negative_predictive_value_sum = 0.0
            true_negative_rate_sum = 0.0
            false_positive_rate_sum = 0.0
            false_negative_rate_sum = 0.0
            proportion_pred_neg_sum = 0.0

            for line in lines:
                precision,recall,negative_predictive_value,true_negative_rate,false_positive_rate,false_negative_rate,proportion_pred_neg,total_value = [float(x) for x in line.split(",")]
                precision_sum += precision*total_value
                recall_sum += recall*total_value
                negative_predictive_value_sum += negative_predictive_value*total_value
                true_negative_rate_sum += true_negative_rate*total_value
                false_positive_rate_sum += false_positive_rate*total_value
                false_negative_rate_sum += false_negative_rate*total_value
                proportion_pred_neg_sum += proportion_pred_neg*total_value
                total_values += total_value
            with open("results.csv", 'a') as results_file:
                results_file.write(f"{model_str},{precision_sum/total_values},{recall_sum/total_values},{negative_predictive_value_sum/total_values},{true_negative_rate_sum/total_values},{false_positive_rate_sum/total_values},{false_negative_rate_sum/total_values},{proportion_pred_neg_sum/total_values},{metric_str},")

    if write_newline:
        with open("results.csv", 'a') as results_file:
            results_file.write("\n")

def process_awsm_stats(model_str, metric_str, write_newline = True):
    if os.path.exists("awsm_stats.csv"):
        with open("awsm_stats.csv", 'r') as stats_file:
            lines = stats_file.readlines()
            total_values = 0
            prop_quantized_attn_score_sum = 0.0
            prop_quantized_sum_exp_sum = 0.0
            prop_quantized_sum = 0.0

            for line in lines:
                prop_quantized_attn_score,prop_quantized_sum_exp,prop_quantized,total_value = [float(x) for x in line.split(",")]
                prop_quantized_attn_score_sum += prop_quantized_attn_score*total_value
                prop_quantized_sum_exp_sum += prop_quantized_sum_exp*total_value
                prop_quantized_sum += prop_quantized*total_value
                total_values += total_value
            with open("results.csv", 'a') as results_file:
                results_file.write(f"{model_str},{prop_quantized_attn_score_sum/total_values},{prop_quantized_sum_exp_sum/total_values},{prop_quantized_sum/total_values},{total_values},{metric_str},")

    if write_newline:
        with open("results.csv", 'a') as results_file:
            results_file.write("\n")
        

def parse_config(config_file):
    with open(config_file, 'r') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError:
            print("Invalid JSON file.")
            return [None]
        return config

def clear_tmp_files():
    if os.path.exists("prediction_stats.csv"):
        os.remove("prediction_stats.csv")
    if os.path.exists("characterization.csv"):
        os.remove("characterization.csv")
    if os.path.exists("awsm_stats.csv"):
        os.remove("awsm_stats.csv")
    if os.path.exists("frequency_stats.csv"):
        os.remove("frequency_stats.csv")
    if os.path.exists("quantization_error.csv"):
        os.remove("quantization_error.csv")
    if os.path.exists("macs.csv"):
        os.remove("macs.csv")
    

def collect_frequency_stats(tensor: Tensor, tensor_name: str, pnlq_config = None):
    if pnlq_config is not None and 'profiling' in pnlq_config and "frequency" in pnlq_config["profiling"] and "pause_profiling" not in pnlq_config:
        num_bins = 256
        if tensor_name not in collect_frequency_stats.params:
            tensor_max = math.ceil(tensor.abs().max().item())
            collect_frequency_stats.params[tensor_name] = (tensor_max, num_bins / tensor_max / 2)
    
        max_value = collect_frequency_stats.params[tensor_name][0]
        scaling = collect_frequency_stats.params[tensor_name][1]
        with open("frequency_stats.csv", 'a') as stats_file:
            stats_file.write(f"{tensor_name},{max_value},{scaling},")
            stats_file.write(f"{torch.sum(torch.where(tensor < (1 / scaling - max_value), 1, 0))},")
            for i in range(1, num_bins - 1):
                stats_file.write(f"{torch.sum(torch.where((tensor >= (i / scaling - max_value)) & (tensor < ((i + 1) / scaling - max_value)), 1, 0))},")
            stats_file.write(f"{torch.sum(torch.where((tensor >= (num_bins - 1) / scaling - max_value), 1, 0))},")
            stats_file.write("\n")

def pause_profiling(pnlq_config):
    pnlq_config["pause_profiling"] = True

def resume_profiling(pnlq_config):
    if "pause_profiling" in pnlq_config:
        del pnlq_config["pause_profiling"]

collect_frequency_stats.params = {}
    
def process_frequency_stats(model_str):
    if os.path.exists("frequency_stats.csv"):
        with open("frequency_stats.csv", 'r') as stats_file:
            lines = stats_file.readlines()
            frequency_tracker = {}

            for line in lines:
                values = line.split(",")
                tensor_name = values[0]
                if tensor_name not in frequency_tracker:
                    frequency_tracker[tensor_name] = {"max": float(values[1]), "scaling": float(values[2]), "bins": [0] * 256}
                tensor_values = [int(x) for x in values[3:-1]]
                for i in range(256):
                    frequency_tracker[tensor_name]["bins"][i] += tensor_values[i]

            with open("frequency_summary.csv", 'a') as frequency_file:
                frequency_file.write("Model,Config,Tensor Name,Max Value,Scaling,Bins\n")
                for tensor_name in frequency_tracker:
                    freqs = ", ".join([str(i) for i in frequency_tracker[tensor_name]['bins']])
                    frequency_file.write(f"{model_str},{tensor_name},{frequency_tracker[tensor_name]['max']},{frequency_tracker[tensor_name]['scaling']},{freqs}\n")

def collect_quantization_error_stats(full_precision, quantized, tensor_name, dim = 1, config = None):
    if config is not None and 'profiling' in config and "quantization_error" in config["profiling"]:
        total_values = full_precision.numel()
        mse = F.mse_loss(quantized, full_precision)
        full_precision = F.log_softmax(full_precision, dim=dim)
        quantized = F.log_softmax(quantized, dim=dim)
        full_precision = full_precision.flatten()
        quantized = quantized.flatten()
        kl_divergence = F.kl_div(quantized, full_precision, reduction='mean', log_target=True)

        with open("quantization_error.csv", 'a') as file:
            file.write(f"{tensor_name},{kl_divergence},{mse},{total_values}\n")

def process_quantization_error_stats(model_str):
    if os.path.exists("quantization_error.csv"):
        with open("quantization_error.csv", 'r') as qe_file:
            qe_tracker = {}
            lines = qe_file.readlines()

            for line in lines:
                tensor_name,kl_divergence,mse,total_values = line.split(",")
                if tensor_name not in qe_tracker:
                    qe_tracker[tensor_name] = {"kl_sum": 0.0, "mse_sum": 0.0, "total_values": 0}
                qe_tracker[tensor_name]["kl_sum"] += float(kl_divergence)
                qe_tracker[tensor_name]["mse_sum"] += float(mse)
                qe_tracker[tensor_name]["total_values"] += int(total_values)
            with open("quantization_error_summary.csv", 'a') as summary_file:
                for tensor_name in qe_tracker:
                    summary_file.write(f"{model_str},{tensor_name},{qe_tracker[tensor_name]['kl_sum'] / qe_tracker[tensor_name]['total_values']},{qe_tracker[tensor_name]['mse_sum'] / qe_tracker[tensor_name]['total_values']}\n")

def setup_profiling(pnlq_config, model, input):
    if pnlq_config is not None and 'profiling' in pnlq_config:
        if "macs" in pnlq_config['profiling']:
            with open ("macs.csv", "a") as f:
                f.write(f"\n{pnlq_config['profiling']}, ")
            if pnlq_config["profiling"] == "macs_all":
                macs = profile_macs(model, input)
                with open ("macs.csv", "a") as f:
                    f.write(f"{macs}, ")
        return True
    else:
        return False


def process_macs_stats(model_str):
    if os.path.exists("macs.csv"):
        with open("macs.csv", 'r') as macs_file:
            lines = macs_file.readlines()
            for line in lines:
                split = line.split(",")
                profiling = split[0]
                macs = [int(x) for x in split[1:-1]]
                with open("macs_summary.csv", 'a') as sum_file:
                    sum_file.write(f"{model_str},{profiling},{sum(macs)}\n")