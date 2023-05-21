class MPT7B4BitConfig:
    name = 'mpt-7b-storywriter-4bit'
    hf_config_name = "OccamRazor/mpt-7b-storywriter-4bit-128g"
    hf_tokenizer_config = "EleutherAI/gpt-neox-20b"
    bits = 4
    groupsize = 128
    max_seq_len = 5 * 2048
    attn_impl = 'torch'
    device_map = "auto"
