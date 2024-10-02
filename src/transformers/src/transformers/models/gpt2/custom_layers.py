import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from ... import activations

bincount_map = {}

threshold_values_to_sweep = [0, -0.002, -0.004, -0.006, -0.008, -0.01, 0, -0.002, -0.004, -0.006, -0.008, -0.01]
# threshold_values_to_sweep = [0, -0.0005, -0.001, -0.0015, -0.002, -0.0025, 0, -0.0005, -0.001, -0.0015, -0.002, -0.0025]

num_bits_to_calculate_to_sweep = [3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
prediction_values_to_sweep = [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05]

fraction_of_computation_to_sweep = [0.3, 0.4, 0.5, 0.6,0.7,1]

class Linear_GELU(nn.Module):
    r"""GELU but logs proportion of negative values.
    """
    __constants__ = ['approximate', 'in_features', 'out_features']
    approximate: str
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU, self).__init__()
        self.approximate = approximate
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
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

    def forward(self, input: Tensor) -> Tensor:
        #collect_frequency_stats(input, "input")
        # collect_frequency_stats(self.weight, "weight")
        # collect_neg_stats(self.bias, "bias")
        input = input @ self.weight.t()
        collect_neg_stats(input, "prod")
        input += self.bias
        collect_neg_stats(input, "prod+bias")
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        return 'approximate={}'.format(repr(self.approximate))

class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Added A2GENT prediction

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        self.idx = 0
        self.max_input_value = 8
        self.max_weight_value = 0.0625
        self.gelu_impl = activations.NewGELUActivation()
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, input):
        # example dims - input.shape:  torch.Size([128, 3136, 128]), self.weight.shape:  torch.Size([512, 128])

        threshold = threshold_values_to_sweep[self.idx] if len(threshold_values_to_sweep) > self.idx else -0.04
        single_value_prediction = prediction_values_to_sweep[self.idx] if len(prediction_values_to_sweep) > self.idx else -0.09
        num_bits_to_calculate = num_bits_to_calculate_to_sweep[self.idx] if len(num_bits_to_calculate_to_sweep) > self.idx else 3
        weight_num_bits_to_calculate = 3
        max_input_value = self.max_input_value
        max_weight_value = 2**torch.round(torch.log2(torch.max(torch.abs(self.weight),dim=-1, keepdim=True).values))
        # print("input.shape: ", input.shape)
        # print("max_input_value.shape: ", max_input_value.shape)
        # print("weight.shape: ", self.weight.shape)
        # print("max_weight_value.shape: ", max_weight_value.shape)

        input_sign_values = signed_quantization_no_shift(input, num_bits_to_calculate, max_input_value)
        weight_sign_values = signed_quantization_no_shift(self.weight, weight_num_bits_to_calculate, max_weight_value)
        bias_sign_value = signed_quantization_no_shift(self.bias, num_bits_to_calculate, max_input_value)

        # input_sign_values = torch.sign(input)
        # weight_sign_values = torch.sign(self.weight)

        output_sign_values = (input_sign_values @ weight_sign_values)/(self.weight.shape[-1])/2**((num_bits_to_calculate+weight_num_bits_to_calculate)/2)

        output_sign_values_rev_quantized = (reverse_signed_quantization_no_shift(input_sign_values, num_bits_to_calculate, max_input_value) @ reverse_signed_quantization_no_shift(weight_sign_values, weight_num_bits_to_calculate, max_weight_value)) + reverse_signed_quantization_no_shift(bias_sign_value, num_bits_to_calculate, max_input_value)

        reverse_quantized_output_sign_values_and_gelu = self.gelu_impl(output_sign_values_rev_quantized)

        pred_neg_values = torch.where(output_sign_values < threshold, 1, 0)

        prod_output = input @ self.weight
        linear_output = prod_output + self.bias
        output = self.gelu_impl(linear_output)

        # torch.set_printoptions(sci_mode=False)
        # print("self.weight.shape", self.weight.shape)
        # print("input.shape", input.shape)
        # print("pred_neg_values.shape", pred_neg_values.shape)
        # print("max input: ", torch.max(input).item(), ", max weight: ", torch.max(self.weight).item())


        # torch.set_printoptions(sci_mode=False)
        # for i in range(pred_neg_values.shape[0]):
        #     for ii in range(pred_neg_values.shape[1]):
        #         for iii in range(pred_neg_values.shape[2]):
        #             if(pred_neg_values[i][ii][iii] == 1):
        #                 if (abs(reverse_quantized_output_sign_values_and_gelu[i][ii][iii].item() - output[i][ii][iii].item()) > 0.02):
        #                     print("not close")
        #                 # else:
        #                 #     print("close")
        #                     print("max input:\n", torch.max(input[i][ii]))
        #                     print("max weight:\n", torch.max(self.weight[iii]))
        #                     print("pred, output: ", reverse_quantized_output_sign_values_and_gelu[i][ii][iii].item(), ", ", output[i][ii][iii].item())

        #                     print("bias:\n", self.bias[iii])
        #                     print("input:\n", input[i][ii].reshape(input.shape[-1]//8, 8))
        #                     print("weight.t():\n", self.weight[iii].t().reshape(self.weight.shape[-1]//8, 8))
        #                     print("input*weight.t():\n", torch.mul(input[i][ii], self.weight[iii].t()).reshape(input.shape[-1]//8, 8))
        #                     print("input@weight.t():\n", input[i][ii]@self.weight[iii].t())
        #                     print("reverse_quant_input:\n", reverse_signed_quantization_no_shift(input_sign_values, num_bits_to_calculate, max_input_value)[i][ii].reshape(input.shape[-1]//8, 8))
        #                     print("reverse_quant_weight:\n", reverse_signed_quantization_no_shift(weight_sign_values, num_bits_to_calculate, max_weight_value)[iii].t().reshape(input.shape[-1]//8, 8))

        #                     print("prod_output:\n", prod_output[i][ii][iii])
        #                     print("input_sign_values:\n", input_sign_values[i][ii].reshape(input_sign_values.shape[-1]//8, 8))
        #                     print("weight_sign_values:\n", weight_sign_values[iii].reshape(weight_sign_values.shape[-1]//8, 8))

        #                     print("input*weight.t():\n", torch.mul(input[i][ii], self.weight[iii].t()).reshape(input.shape[-1]//8, 8))
        #                     print("reverse signed input*weight.():\n", torch.mul(reverse_signed_quantization_no_shift(input_sign_values, num_bits_to_calculate, max_input_value)[i][ii], reverse_signed_quantization_no_shift(weight_sign_values, num_bits_to_calculate, max_weight_value)[iii].t()).reshape(input.shape[-1]//8, 8))
        #                     print("reverse_output_sign_values:\n", output_sign_values_rev_quantized[i][ii][iii])

        size_out = input.size()[:-1] + (self.nf,)
        output = torch.where(output_sign_values < threshold, reverse_quantized_output_sign_values_and_gelu, output).view(size_out)


        predicted = pred_neg_values
        actual = torch.where(prod_output >= 0, 0, 1)

        # true_positives = torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 1, 1, 0))

        # false_positives = torch.mul(torch.where(predicted != actual, 1, 0), torch.where(predicted == 1, 1, 0))
        # print("input.shape: ", input.shape)
        # print("weight.shape: ", self.weight.shape)
        # print("predicted.shape: ", predicted.shape)
        # print("actual.shape: ", actual.shape)
        # print("false_positives.shape: ", false_positives.shape)

        # print("False positives:")
        # torch.set_printoptions(sci_mode=False)

        # for i in range(false_positives.shape[0]):
        #     for ii in range(false_positives.shape[1]):
        #         for iii in range(false_positives.shape[2]):
        #             if false_positives[i][ii][iii] == 1:
        #                 print("input:\n", input[i][ii].reshape(input.shape[-1]//8, 8))
        #                 print("weight:\n", self.weight[iii].reshape(self.weight.shape[-1]//8, 8))
        #                 print("input*weight:\n", torch.mul(input[i][ii], self.weight[iii]).reshape(input.shape[-1]//8, 8))
        #                 print("prod_output:\n", prod_output[i][ii][iii])
        #                 print("input_sign_values:\n", input_sign_values[i][ii].reshape(input_sign_values.shape[-1]//8, 8))
        #                 print("weight_sign_values:\n", weight_sign_values[iii].reshape(weight_sign_values.shape[-1]//8, 8))
        #                 print("signed input*weight:\n", torch.mul(input_sign_values[i][ii], weight_sign_values[iii]).reshape(input.shape[-1]//8, 8))
        #                 print("output_sign_values:\n", output_sign_values[i][ii][iii])

        collect_prediction_stats(predicted, actual, threshold, single_value_prediction, num_bits_to_calculate)

        return output

class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction1(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, nf, nx):
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction1, self).__init__(nf, nx)
        self.idx = 1

class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction2(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, nf, nx):
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction2, self).__init__(nf, nx)
        self.idx = 2


class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction3(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, nf, nx) -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction3, self).__init__(nf, nx)
        self.idx = 3

class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction4(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, nf, nx) -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction4, self).__init__(nf, nx)
        self.idx = 4


class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction5(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, nf, nx) -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction5, self).__init__(nf, nx)
        self.idx = 5


class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction6(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, nf, nx) -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction6, self).__init__(nf, nx)
        self.idx = 6



class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction7(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, nf, nx) -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction7, self).__init__(nf, nx)
        self.idx = 7


class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction8(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, nf, nx) -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction8, self).__init__(nf, nx)
        self.idx = 8


class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction9(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, nf, nx) -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction9, self).__init__(nf, nx)
        self.idx = 9


class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction10(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, nf, nx) -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction10, self).__init__(nf, nx)
        self.idx = 10

class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction11(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, nf, nx) -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction11, self).__init__(nf, nx)
        self.idx = 11

def collect_prediction_stats(predicted, actual, threshold = 0, single_value_prediction = 0, num_bits_to_calculate = 0):
    precision = torch.sum(torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 1, 1, 0))) / torch.sum(torch.where(predicted == 1, 1, 0))
    recall = torch.sum(torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 1, 1, 0))) / torch.sum(torch.where(actual == 1, 1, 0))
    negative_predictive_value = torch.sum(torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 0, 1, 0))) / torch.sum(torch.where(predicted == 0, 1, 0))
    true_negative_rate = torch.sum(torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 0, 1, 0))) / torch.sum(torch.where(actual == 0, 1, 0))
    false_positive_rate = torch.sum(torch.mul(torch.where(predicted != actual, 1, 0), torch.where(predicted == 1, 1, 0))) / torch.sum(torch.where(actual == 0, 1, 0))
    false_negative_rate = torch.sum(torch.mul(torch.where(predicted != actual, 1, 0), torch.where(predicted == 0, 1, 0))) / torch.sum(torch.where(actual == 1, 1, 0))

    total_values = actual.numel()
    proportion_pred_neg = torch.sum(torch.where(predicted == 1, 1, 0)) / total_values

    neg = torch.sum(actual)

    with open("mispredictions.csv", 'a') as results_file:
        results_file.write(f"{neg},{precision},{recall},{negative_predictive_value},{true_negative_rate},{false_positive_rate},{false_negative_rate},{proportion_pred_neg},{total_values},{threshold},{single_value_prediction},{num_bits_to_calculate}\n")

def collect_frequency_stats(matrix, tag):
    matrix = F.relu((matrix*10).to(torch.int32) + 200)
    if tag not in bincount_map:
        bincount_map[tag] = torch.bincount(matrix.flatten(), minlength=400)
    else:
        bincount_map[tag] += torch.bincount(matrix.flatten(), minlength=400)

    with open(f"bincount_{tag}.csv", 'a') as bincount_file:
        for tag, bincount in bincount_map.items():
            bincount_file.write(f"{tag}, "+ ",".join(str(int(x)) for x in bincount)+ "\n")

def collect_neg_stats(matrix, tag):
    neg = torch.sum(torch.where(matrix < 0, 1, 0))

    total = matrix.numel()
    with open(f"neg_values_{tag}.csv", 'a') as bincount_file:
        bincount_file.write(f"{tag}, {neg}, {total}"+ "\n")


def signed_quantization_no_shift(matrix, num_non_signbits, max_value):
    num_values_in_quantization = 2**num_non_signbits
    max_value_in_quantization = num_values_in_quantization - 1
    return torch.sign(matrix) * torch.min(torch.round((torch.abs(matrix))/max_value * num_values_in_quantization) + 1, torch.tensor(max_value_in_quantization).cuda())

def reverse_signed_quantization_no_shift(matrix, num_non_signbits, max_value):
    num_values_in_quantization = 2**num_non_signbits
    return torch.sign(matrix) * (torch.abs(matrix) - 1) / num_values_in_quantization * max_value

LINEAR_GELU_FN = {
    0 : Linear_GELU,
    5 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction,
    6 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction1,
    7 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction2,
    8 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction3,
    9 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction4,
    10 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction5,
    11 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction6,
    12 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction7,
    13 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction8,
    14 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction9,
    15 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction10,
    16 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction11,
}