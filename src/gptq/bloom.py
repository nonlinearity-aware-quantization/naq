import math
import time

import torch
import torch.nn as nn
import transformers

from gptq import * 
from modelutils import *
from quant import *
import json
import pnlq
import os


def get_bloom(model, pnlq_config):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import BloomForCausalLM
    mod_config = transformers.BloomConfig.from_pretrained(model)
    mod_config.pnlq_config = pnlq_config
    model = BloomForCausalLM.from_pretrained(model, torch_dtype='auto', config = mod_config)
    model.seqlen = 2048
    return model

@torch.no_grad()
def bloom_sequential(model, dataloader, dev, means=None, stds=None):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(i, name)
            print('Quantizing ...')
            gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize)
            quantizers['transformer.h.%d.%s' % (i, name)] = gptq[name].quantizer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers

@torch.no_grad()
def bloom_eval(model, testenc, dev):
    print('Evaluation...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        layers[i] = layer.cpu() 
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.transformer.ln_f = model.transformer.ln_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        hidden_states = model.transformer.ln_f(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    with open("results.csv", 'a') as results_file:
        results_file.write(f"{ppl.item():3f},\n")

    model.config.use_cache = use_cache
    return ppl.item()


def bloom_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

def main_fn(args, pnlq_config):
    pnlq.clear_tmp_files()
    pnlq.pause_profiling(pnlq_config)
    model = get_bloom(args.model, pnlq_config)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = bloom_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    datasets = ['wikitext2', 'ptb', 'c4'] 
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        pnlq.resume_profiling(pnlq_config)
        ppl = bloom_eval(model, testloader, DEV)

        model_str = f"{args.model},{dataset},{json.dumps(pnlq_config) if pnlq_config != None else 'None'}"
        print(model_str)

        if pnlq_config != None:
            pnlq.process_prediction_stats(model_str,f"{ppl:.2f}", write_newline=False)
            pnlq.process_awsm_stats(model_str, f"{ppl:.2f}")
            pnlq.process_frequency_stats(model_str)
            pnlq.process_quantization_error_stats(model_str)
            pnlq.process_macs_stats(model_str)
        if os.path.exists("frequency_summary.csv"):
            os.rename("frequency_summary.csv", f"frequency_summary_{dataset}.csv")

    if args.save:
        bloom_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='BLOOM model to load; pass `bigscience/bloom-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval'
    )
    parser.add_argument("--pnlq_config_file", type=str, required=False, help='file with pnlq configuration json')


    args = parser.parse_args()


    if hasattr(args, "pnlq_config_file") and args.pnlq_config_file is not None:
        pnlq_configs = pnlq.parse_config(args.pnlq_config_file)
    else:
        pnlq_configs = [{}]

    for pnlq_config in pnlq_configs:
        main_fn(args, pnlq_config)