# naq

Install [GPTQ](https://github.com/IST-DASLab/gptq) and [PTQ4ViT](https://github.com/hahnyuan/PTQ4ViT) as normal except install the transformer package for GPTQ using `src/transformer` and the timm package for PTQ4ViT using `src/timm-0.6.5m`. Use `pip install -e /path/to/pkg`

Sample commands to run experiments:
- `python bloom.py bigscience/bloom-1b1 c4 --wbits 4 --pnlq_config_file pnlq_config_bloom.json`
- `python opt.py facebook/opt-125m c4 --wbits 4 --pnlq_config_file pnlq_config_opt.json`
- `python example/test_all.py --pnlq_config pnlq_config.json`
