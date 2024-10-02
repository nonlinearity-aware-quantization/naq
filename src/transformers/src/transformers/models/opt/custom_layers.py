import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

bincount_map = {}

threshold_values_to_sweep = [0, -0.01, -0.02, -0.03, -0.04, -0.05, 0, -0.01, -0.02, -0.03, -0.04, -0.05]
num_bits_to_calculate_to_sweep = [4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
prediction_values_to_sweep = [-0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05]

fraction_of_computation_to_sweep = [0.3, 0.4, 0.5, 0.6,0.7,1]


class Linear_Logging(nn.Module):
    r"""GELU but logs proportion of negative values.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
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
        num_neg_values = torch.sum(torch.where(input >= 0, 0, 1))
        num_values = input.numel()
        with open("neg_values_act.csv", 'a') as results_file:
            results_file.write(f"{num_neg_values},{num_values},\n")
        num_neg_values = torch.sum(torch.where(self.weight >= 0, 0, 1))
        num_values = self.weight.numel()
        with open("neg_values_weight.csv", 'a') as results_file:
            results_file.write(f"{num_neg_values},{num_values},\n")
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

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

class Linear_GELU_Predict_Negative(Linear_GELU):
    def forward(self, input: Tensor) -> Tensor:
        # example dims - input.shape:  torch.Size([128, 3136, 128]), self.weight.shape:  torch.Size([512, 128])
        output_shape = list(input.shape[:-1]) +[self.weight.shape[0]]
        pred_neg_values = torch.zeros(tuple(output_shape), device=input.device) + 1
        linear_output = input @ self.weight.t() + self.bias
        output = F.gelu(linear_output, approximate=self.approximate)
        expected = pred_neg_values
        actual = torch.where(linear_output >= 0, 0, 1)

        collect_prediction_stats(expected, actual, 0)

        return output

class Linear_GELU_Single_Threshold_Prediction(Linear_GELU):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_Single_Threshold_Prediction, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.threshold = 0.0

    def forward(self, input: Tensor) -> Tensor:
        # example dims - input.shape:  torch.Size([128, 3136, 128]), self.weight.shape:  torch.Size([512, 128])
        output_shape = list(input.shape[:-1]) +[self.weight.shape[0]]
        # pred_neg_values = torch.zeros(tuple(output_shape), device=input.device) + 1

        std = torch.std(input, unbiased = False)

        pred_neg_values = torch.where(input.mean(dim=-1, keepdim=True) < self.threshold, 1, 0)

        linear_output = input @ self.weight.t() + self.bias
        output = F.gelu(linear_output, approximate=self.approximate)

        if self.threshold == 0.0:
            self.threshold = 3 * std

        if torch.sum(torch.where(linear_output >= 0, 0, 1) - pred_neg_values) > 0:
            # actual output is more negative than predicted
            self.threshold += std * 0.1
        else:
            # actual output is less negative than predicted
            self.threshold -= std * 0.1

        expected = pred_neg_values
        actual = torch.where(linear_output >= 0, 0, 1)

        collect_prediction_stats(expected, actual, self.threshold)

        return output

class Linear_GELU_MeanMeanThreshold(Linear_GELU):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_MeanMeanThreshold, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.threshold = 0.0


    def forward(self, input: Tensor) -> Tensor:
        # example dims - input.shape:  torch.Size([128, 3136, 128]) [batch_size, dim1, dim2], self.weight.shape:  torch.Size([512, 128])

        input_mean = torch.mean(input, -1,  keepdims=True)
        weight_mean = torch.mean(self.weight, -1,  keepdims=True)

        input_std = torch.std(input, unbiased = False)
        weight_std = torch.std(self.weight, unbiased = False)
        std = input_std * weight_std

        mean_output = input_mean @ weight_mean.t() + self.bias

        pred_neg_values = torch.where(mean_output < self.threshold, 1, 0)

        linear_output = input @ self.weight.t() + self.bias
        output = F.gelu(linear_output, approximate=self.approximate)

        assert linear_output.shape == mean_output.shape

        if torch.sum(torch.where(linear_output >= 0, 0, 1) - pred_neg_values) > 0:
            # actual output is more negative than predicted
            self.threshold -= std * 0.1
        else:
            # actual output is less negative than predicted
            self.threshold += std * 0.1

        mispredictions = torch.sum(torch.where(torch.where(linear_output >= 0, 0, 1) != pred_neg_values, 1, 0))

        expected = pred_neg_values
        actual = torch.where(linear_output >= 0, 0, 1)

        collect_prediction_stats(expected, actual, self.threshold)

        return output

class Linear_GELU_PerWeightVectorThreshold(Linear_GELU):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_PerWeightVectorThreshold, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.threshold = torch.zeros((1, 1, out_features)).cuda()


    def forward(self, input: Tensor) -> Tensor:
        # example dims - input.shape:  torch.Size([128, 3136, 128]) [batch_size, dim1, dim2], self.weight.shape:  torch.Size([512, 128])

        input_mean = torch.mean(input, -1,  keepdims=True)

        delta_mean_threshold = input_mean - self.threshold

        pred_neg_values = torch.where(delta_mean_threshold <= 0, 1, 0)

        linear_output = input @ self.weight.t()
        output = F.gelu(linear_output + self.bias, approximate=self.approximate)

        expected = pred_neg_values
        actual = torch.where(linear_output >= 0, 0, 1)

        # threshold_adjustment = torch.where(actual != expected, delta_mean_threshold, 0)
        # self.threshold += torch.mean(threshold_adjustment, dim=(0,1))

        collect_prediction_stats(expected, actual, torch.mean(self.threshold))

        return output

class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction(Linear_GELU):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.idx = 0
        self.max_input_value = 8
        self.max_weight_value = 0.0625

    def forward(self, input: Tensor) -> Tensor:
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

        proportion_weight_zero = torch.sum(torch.where(self.weight == 0, 1, 0), dim=-1)/weight_sign_values.numel()

        # input_sign_values = torch.sign(input)
        # weight_sign_values = torch.sign(self.weight)

        output_sign_values = (input_sign_values @ weight_sign_values.t())/(self.weight.shape[-1])/2**((num_bits_to_calculate+weight_num_bits_to_calculate)/2)

        output_sign_values_rev_quantized = (reverse_signed_quantization_no_shift(input_sign_values, num_bits_to_calculate, max_input_value) @ reverse_signed_quantization_no_shift(weight_sign_values, weight_num_bits_to_calculate, max_weight_value).t()) + reverse_signed_quantization_no_shift(bias_sign_value, num_bits_to_calculate, max_input_value)

        reverse_quantized_output_sign_values_and_gelu = F.relu(output_sign_values_rev_quantized)

        pred_neg_values = torch.where(output_sign_values < threshold, 1, 0)

        prod_output = input @ self.weight.t()
        linear_output = prod_output + self.bias
        output = F.relu(linear_output)
        # print("output.shape: ", output.shape)

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

        output = torch.where(output_sign_values < threshold, reverse_quantized_output_sign_values_and_gelu, output)


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

        collect_prediction_stats(predicted, actual, threshold, single_value_prediction, num_bits_to_calculate, proportion_weight_zero)

        return output

class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction1(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction1, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.idx = 1

class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction2(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction2, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.idx = 2


class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction3(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction3, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.idx = 3

class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction4(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction4, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.idx = 4


class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction5(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction5, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.idx = 5


class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction6(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction6, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.idx = 6



class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction7(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction7, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.idx = 7


class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction8(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction8, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.idx = 8


class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction9(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction9, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.idx = 9


class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction10(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction10, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.idx = 10

class Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction11(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction11, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.idx = 11

class Linear_GELU_Characterization(Linear_GELU):
    r"""Characterization
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_Characterization, self).__init__(in_features, out_features, bias, device, dtype, approximate)

    def forward(self, input: Tensor) -> Tensor:
        input = input @ self.weight.t()
        # collect_bias_characterization_stats(input, self.bias)
        # collect_characterization_stats(input, self.weight, self.bias)
        collect_input_characterization_stats(input)
        input += self.bias
        return F.relu(input)

    def extra_repr(self) -> str:
        return 'approximate={}'.format(repr(self.approximate))

class Linear_GELU_SnaPEA(Linear_GELU):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SnaPEA, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.sorting_type = None

    def forward(self, input: Tensor) -> Tensor:
        # example dims - input.shape:  torch.Size([128, 3136, 128]), self.weight.shape:  torch.Size([512, 128])

        first_disagreement_from_the_end_tracker_pos = torch.zeros((self.weight.shape[1]+1,), device=input.device)
        first_disagreement_from_the_end_tracker_neg = torch.zeros((self.weight.shape[1]+1,), device=input.device)

        input_reshaped = input.unsqueeze(2)
        weight_reshaped = self.weight.unsqueeze(0)

        for sample in range(input.shape[0]):
            prod = torch.mul(input_reshaped[sample], weight_reshaped)

            if self.sorting_type == "weight":
                _, indices = torch.sort(weight_reshaped, dim = 2,descending=True)
                prod = torch.cat([torch.gather(prod[x].unsqueeze(0), 2, indices) for x in range(prod.shape[0])], dim=0)
            if self.sorting_type == "input":
                _, indices = torch.sort(input_reshaped[sample], dim = 2,descending=True)
                prod = torch.cat([torch.gather(prod[:,x].unsqueeze(1), 2, indices) for x in range(prod.shape[1])], dim=1)
            assert(prod.shape == (input.shape[1], self.weight.shape[0], self.weight.shape[1]))
            
            prod = torch.cumsum(prod, dim=2)

            prod = torch.cumsum(torch.flip(torch.sign(prod), dims=[2]), dim=2)/torch.arange(1, self.weight.shape[1]+1).unsqueeze(0).unsqueeze(0).cuda()
            prod = torch.where(torch.abs(prod) < 0.99999, 1, 0)
            # last element disagrees to avoid argmax return first element if no disagreements
            prod = torch.cat([prod, torch.ones((input.shape[1], self.weight.shape[0], 1), device=input.device)], dim=2)
            first_disagreement_from_the_end = torch.argmax(prod, dim = 2)
            assert(first_disagreement_from_the_end.shape == (input.shape[1], self.weight.shape[0]))

            # if non-positive, then set index to weight.shape[1]+1
            first_disagreement_from_the_end_pos = torch.where(input[sample] @ self.weight.t() > 0, first_disagreement_from_the_end, self.weight.shape[1] + 1)
            # if non-negative, then set index to weight.shape[1]+1
            first_disagreement_from_the_end_neg = torch.where(input[sample] @ self.weight.t() < 0, first_disagreement_from_the_end, self.weight.shape[1] + 1)
            # need to ignore indices set to weight.shape[1]+1

            bincount_pos = torch.bincount(first_disagreement_from_the_end_pos.flatten(), minlength=self.weight.shape[1]+2)
            bincount_neg = torch.bincount(first_disagreement_from_the_end_neg.flatten(), minlength=self.weight.shape[1]+2)

            first_disagreement_from_the_end_tracker_pos += bincount_pos[:-1]
            first_disagreement_from_the_end_tracker_neg += bincount_neg[:-1]
            # only run this loop once
            break

        collect_first_disagreement_from_end_stat(first_disagreement_from_the_end_tracker_pos, "_pos")
        collect_first_disagreement_from_end_stat(first_disagreement_from_the_end_tracker_neg, "_neg")

        linear_output = input @ self.weight.t() + self.bias
        output = F.gelu(linear_output, approximate=self.approximate)

        return output

class Linear_GELU_SnaPEA_WeightSorted(Linear_GELU_SnaPEA):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SnaPEA_WeightSorted, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.sorting_type = "weight"

class Linear_GELU_SnaPEA_InputSorted(Linear_GELU_SnaPEA):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SnaPEA_InputSorted, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.sorting_type = "input"

class Linear_GELU_SnaPEA_Prediction0(Linear_GELU):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SnaPEA_Prediction0, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.fraction_of_computation_to_compute = fraction_of_computation_to_sweep[0] if len(fraction_of_computation_to_sweep) > 0 else 1/2

    def forward(self, input: Tensor) -> Tensor:
        # example dims - input.shape:  torch.Size([128, 3136, 128]), self.weight.shape:  torch.Size([512, 128])

        input_reshaped = input.unsqueeze(2)
        weight_reshaped = self.weight.unsqueeze(0)

        sample = 0
        prod = torch.mul(input_reshaped[sample], weight_reshaped)

        # weight sorting
        _, indices = torch.sort(weight_reshaped, dim = 2,descending=True)
        prod = torch.cat([torch.gather(prod[x].unsqueeze(0), 2, indices) for x in range(prod.shape[0])], dim=0)
        assert(prod.shape == (input.shape[1], self.weight.shape[0], self.weight.shape[1]))
        
        prod = torch.cumsum(prod, dim=2)
        linear_output = input @ self.weight.t() + self.bias

        amount_of_computation = int(input.shape[0]*self.fraction_of_computation_to_compute)

        computed_values = prod[:,:,:amount_of_computation]

        partial_val = computed_values[:,:,-1] # + self.bias

        negative_sign_consistency = torch.where(torch.sum(torch.where(computed_values < 0, 1, 0))/amount_of_computation>0.9999,1,0)
        pred = torch.where(partial_val < 0, 1, 0) * torch.where(negative_sign_consistency == 1, 1, 0)

        linear_output = input @ self.weight.t() + self.bias
        actual = torch.where(linear_output[0] < 0, 1, 0)

        collect_prediction_stats(pred, actual, 0, 0)

        output = F.gelu(linear_output, approximate=self.approximate)

        return output

class Linear_GELU_SnaPEA_Prediction1(Linear_GELU_SnaPEA_Prediction0):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SnaPEA_Prediction1, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.fraction_of_computation_to_compute = fraction_of_computation_to_sweep[1] if len(fraction_of_computation_to_sweep) > 1 else 1/2


class Linear_GELU_SnaPEA_Prediction2(Linear_GELU_SnaPEA_Prediction0):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SnaPEA_Prediction2, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.fraction_of_computation_to_compute = fraction_of_computation_to_sweep[2] if len(fraction_of_computation_to_sweep) > 2 else 1/2

class Linear_GELU_SnaPEA_Prediction3(Linear_GELU_SnaPEA_Prediction0):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SnaPEA_Prediction3, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.fraction_of_computation_to_compute = fraction_of_computation_to_sweep[3] if len(fraction_of_computation_to_sweep) > 3 else 1/2

class Linear_GELU_SnaPEA_Prediction4(Linear_GELU_SnaPEA_Prediction0):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SnaPEA_Prediction4, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.fraction_of_computation_to_compute = fraction_of_computation_to_sweep[4] if len(fraction_of_computation_to_sweep) > 4 else 1/2

class Linear_GELU_SnaPEA_Prediction5(Linear_GELU_SnaPEA_Prediction0):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_SnaPEA_Prediction5, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.fraction_of_computation_to_compute = fraction_of_computation_to_sweep[5] if len(fraction_of_computation_to_sweep) > 5 else 1/2

class Linear_GELU_BitSerial(Linear_GELU):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_BitSerial, self).__init__(in_features, out_features, bias, device, dtype, approximate)
        self.num_bits = 1

    def forward(self, input: Tensor) -> Tensor:
        # example dims - input.shape:  torch.Size([128, 3136, 128]), self.weight.shape:  torch.Size([512, 128])

        quantization_level_output = convert_to_quantization_level(input, 256, 128) @ convert_to_quantization_level(self.weight.t(), 256, 8) + convert_to_quantization_level(self.bias, 256, 16)

        pred_neg_values = torch.where(quantization_level_output < 0, 1, 0)
        linear_output = input @ self.weight.t() + self.bias
        expected = pred_neg_values
        actual = torch.where(linear_output >= 0, 0, 1)
        collect_prediction_stats(expected, actual, 0, 0)

        output = F.gelu(linear_output, approximate=self.approximate)
        return output

class Linear_GELU_NegOnlyBias(Linear_GELU):
    r"""Bias is ReLU'd"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None, approximate: str = 'none') -> None:
        super(Linear_GELU_NegOnlyBias, self).__init__(in_features, out_features, bias, device, dtype, approximate)

    def forward(self, input: Tensor) -> Tensor:
        input = input @ self.weight.t()
        input += -F.relu(-self.bias)
        return F.gelu(input, approximate=self.approximate)

def collect_characterization_stats(prod, bias):
    abs_bias = torch.abs(bias)
    abs_prod = torch.abs(prod)
    bias_dominates = torch.where(abs_bias > abs_prod, 1, 0)
    bias_really_dominates = torch.where((abs_bias - abs_prod) > 3.4662, 1, 0)
    prod_dominates = torch.where(abs_prod > abs_bias, 1, 0)
    sum_bias_dominates = torch.sum(bias_dominates)
    sum_bias_really_dominates = torch.sum(bias_really_dominates)
    sum_prod_dominates = torch.sum(prod_dominates)
    min_bias_dominates = torch.min(torch.where(bias_dominates==1, abs_bias, 1000))
    max_bias_not_dominates = torch.max(torch.where(prod_dominates==1, abs_bias, 0))
    neg_values = torch.sum(torch.where(prod+bias<=0, 1, 0))

    total_values = prod.numel()

    with open("characterization.csv", 'a') as results_file:
        results_file.write(f"{neg_values},{sum_bias_dominates},{sum_bias_really_dominates},{sum_prod_dominates},{min_bias_dominates},{max_bias_not_dominates},{total_values}\n")

def collect_input_characterization_stats(input):
    abs_input = torch.abs(input)
    max_input = torch.max(abs_input)

    with open("characterization.csv", 'a') as results_file:
        results_file.write(f"{max_input}\n")

def collect_bias_characterization_stats(prod, bias):
    abs_bias = torch.abs(bias)
    abs_prod = torch.abs(prod)
    bias_dominates = torch.where(abs_bias > abs_prod, 1, 0)
    bias_positive = torch.where(bias > 0, 1, 0)
    prod_negative = torch.where(prod < 0, 1, 0)

    positive_bias_dominates_negative_prod = torch.sum(torch.mul(torch.mul(bias_dominates, prod_negative), bias_positive))

    total_values = prod.numel()

    with open("characterization.csv", 'a') as results_file:
        results_file.write(f"{positive_bias_dominates_negative_prod},{total_values}\n")


def collect_prediction_stats(predicted, actual, threshold = 0, single_value_prediction = 0, num_bits_to_calculate = 0, proportion_weight_zero = None):
    precision = torch.sum(torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 1, 1, 0))) / torch.sum(torch.where(predicted == 1, 1, 0))
    recall = torch.sum(torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 1, 1, 0))) / torch.sum(torch.where(actual == 1, 1, 0))
    negative_predictive_value = torch.sum(torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 0, 1, 0))) / torch.sum(torch.where(predicted == 0, 1, 0))
    true_negative_rate = torch.sum(torch.mul(torch.where(predicted == actual, 1, 0), torch.where(predicted == 0, 1, 0))) / torch.sum(torch.where(actual == 0, 1, 0))
    false_positive_rate = torch.sum(torch.mul(torch.where(predicted != actual, 1, 0), torch.where(predicted == 1, 1, 0))) / torch.sum(torch.where(actual == 0, 1, 0))
    false_negative_rate = torch.sum(torch.mul(torch.where(predicted != actual, 1, 0), torch.where(predicted == 0, 1, 0))) / torch.sum(torch.where(actual == 1, 1, 0))

    total_values = actual.numel()
    proportion_pred_neg = torch.sum(torch.where(predicted == 1, 1, 0)) / total_values

    neg = torch.sum(actual)

    if proportion_weight_zero is not None:
        average_non_predicted_sparsity = torch.sum(torch.where(predicted == 0, 1, 0) * proportion_weight_zero) / torch.sum(torch.where(predicted == 0, 1, 0))

    with open("mispredictions.csv", 'a') as results_file:
        results_file.write(f"{neg},{precision},{recall},{negative_predictive_value},{true_negative_rate},{false_positive_rate},{false_negative_rate},{proportion_pred_neg},{total_values},{threshold},{single_value_prediction},{num_bits_to_calculate},{average_non_predicted_sparsity}\n")

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


def collect_first_disagreement_from_end_stat(first_diagreement_tracker, tag):
    with open("fdfe/first_disagreement_from_end" + tag + ".csv", 'a') as results_file:
        results_file.write(f"{first_diagreement_tracker.shape[0]}," + ",".join(str(int(x)) for x in first_diagreement_tracker)+ "\n")

def extract_signed_exponent(matrix, bias = 3):
    return torch.sign(matrix) * (torch.floor(torch.log2(torch.abs(matrix))) + 3)

def convert_to_quantization_level(matrix, num_levels, max_val):
    return torch.round(matrix / max_val * num_levels) * max_val / num_levels

def signed_quantization(matrix, num_non_signbits, max_value):
    num_values_in_quantization = 2**num_non_signbits
    quantization_error = max_value / (num_values_in_quantization * 2)
    return torch.sign(matrix) * torch.min(torch.round((torch.abs(matrix)-quantization_error)/max_value * num_values_in_quantization + 1), torch.tensor(num_values_in_quantization).cuda())

def signed_quantization_no_shift(matrix, num_non_signbits, max_value):
    num_values_in_quantization = 2**num_non_signbits
    max_value_in_quantization = num_values_in_quantization - 1
    return torch.sign(matrix) * torch.min(torch.round((torch.abs(matrix))/max_value * num_values_in_quantization) + 1, torch.tensor(max_value_in_quantization).cuda())

def reverse_signed_quantization(matrix, num_non_signbits, max_value):
    num_values_in_quantization = 2**num_non_signbits
    quantization_error = max_value / (num_values_in_quantization * 2)
    return torch.sign(matrix) * (torch.abs(matrix)-1) / num_values_in_quantization * max_value + quantization_error

def reverse_signed_quantization_no_shift(matrix, num_non_signbits, max_value):
    num_values_in_quantization = 2**num_non_signbits
    return torch.sign(matrix) * (torch.abs(matrix) - 1) / num_values_in_quantization * max_value

LINEAR_GELU_FN = {
    0 : Linear_GELU,
    1 : Linear_GELU_Predict_Negative,
    2 : Linear_GELU_Single_Threshold_Prediction,
    3 : Linear_GELU_MeanMeanThreshold,
    4 : Linear_GELU_PerWeightVectorThreshold,
    5 : Linear_GELU_Characterization,
    6 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction,
    7 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction1,
    8 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction2,
    9 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction3,
    10 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction4,
    11 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction5,
    12 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction6,
    13 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction7,
    14 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction8,
    15 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction9,
    16 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction10,
    17 : Linear_GELU_SignedBitMaskMultiplicationThresholdPrediction11,
    18 : Linear_GELU_SnaPEA,
    19 : Linear_GELU_SnaPEA_WeightSorted,
    20 : Linear_GELU_SnaPEA_InputSorted,
    21 : Linear_GELU_SnaPEA_Prediction0,
    22 : Linear_GELU_SnaPEA_Prediction1,
    23 : Linear_GELU_SnaPEA_Prediction2,
    24 : Linear_GELU_SnaPEA_Prediction3,
    25 : Linear_GELU_SnaPEA_Prediction4,
    26 : Linear_GELU_SnaPEA_Prediction5,
    27 : Linear_GELU_BitSerial,
    28 : Linear_GELU_NegOnlyBias,
}