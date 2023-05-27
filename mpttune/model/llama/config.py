class LLama7B4BitConfig:
    name = 'llama-7b-4bit'
    hf_config_name = "decapoda-research/llama-7b-hf"
    hf_tokenizer_config = "huggyllama/llama-13b"
    bits = 4
    groupsize = 128
    max_seq_len = 2048
    device_map = "auto"


class LLama13B4BitConfig:
    name = 'llama-13b-4bit'
    hf_config_name = "decapoda-research/llama-13b-hf"
    hf_tokenizer_config = "huggyllama/llama-13b"
    bits = 4
    groupsize = 128
    max_seq_len = 2048
    device_map = "auto"


class LLama30B4BitConfig:
    name = 'llama-30b-4bit'
    hf_config_name = "decapoda-research/llama-30b-hf"
    hf_tokenizer_config = "huggyllama/llama-13b"
    bits = 4
    groupsize = 128
    max_seq_len = 2048
    device_map = "auto"


class LLama65B4BitConfig:
    name = 'llama-65b-4bit'
    hf_config_name = "decapoda-research/llama-65b-hf"
    hf_tokenizer_config = "huggyllama/llama-13b"
    bits = 4
    groupsize = 128
    max_seq_len = 2048
    device_map = "auto"
