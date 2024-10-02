import os    

def process_mispredictions():
    with open("mispredictions.csv", 'r') as neg_values_file:
        lines = neg_values_file.readlines()
        total_values = 0
        precision_sum = 0.0
        recall_sum = 0.0
        negative_predictive_value_sum = 0.0
        true_negative_rate_sum = 0.0
        false_positive_rate_sum = 0.0
        false_negative_rate_sum = 0.0
        proportion_pred_neg_sum = 0.0
        threshold = 0.0
        single_value_prediction = 0.0
        num_bits_to_calculate = 0.0
        neg_values_sum = 0.0
        average_non_predicted_sparsity_sum = 0.0

        for line in lines:
            neg_values,precision,recall,negative_predictive_value,true_negative_rate,false_positive_rate,false_negative_rate,proportion_pred_neg,total_value,threshold,single_value_prediction,num_bits_to_calculate, average_non_predicted_sparsity = [float(x) for x in line.split(",")]
            neg_values_sum += neg_values
            precision_sum += precision*total_value
            recall_sum += recall*total_value
            negative_predictive_value_sum += negative_predictive_value*total_value
            true_negative_rate_sum += true_negative_rate*total_value
            false_positive_rate_sum += false_positive_rate*total_value
            false_negative_rate_sum += false_negative_rate*total_value
            proportion_pred_neg_sum += proportion_pred_neg*total_value
            average_non_predicted_sparsity_sum += average_non_predicted_sparsity*total_value
            total_values += total_value
        with open("results.csv", 'a') as results_file:
            results_file.write(f"{neg_values_sum/total_values},{precision_sum/total_values},{recall_sum/total_values},{negative_predictive_value_sum/total_values},{true_negative_rate_sum/total_values},{false_positive_rate_sum/total_values},{false_negative_rate_sum/total_values},{proportion_pred_neg_sum/total_values},{threshold},{single_value_prediction},{num_bits_to_calculate},{average_non_predicted_sparsity_sum/total_values},")


def process_frequency_stats(model_str):
    if os.path.exists("frequency_stats_testing.csv"):
        with open("frequency_stats_testing.csv", 'r') as stats_file:
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

            with open("frequency_summary_testing.csv", 'a') as frequency_file:
                frequency_file.write("Model,Config,Tensor Name,Max Value,Scaling,Bins\n")
                for tensor_name in frequency_tracker:
                    freqs = ", ".join([str(i) for i in frequency_tracker[tensor_name]['bins']])
                    frequency_file.write(f"{model_str},{tensor_name},{frequency_tracker[tensor_name]['max']},{frequency_tracker[tensor_name]['scaling']},{freqs}\n")

process_frequency_stats("gptq")