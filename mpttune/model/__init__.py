from mpttune.model.mpt.config import MPT7B4BitConfig


MODEL_CONFIGS = {
    MPT7B4BitConfig.name: MPT7B4BitConfig,
}


def load_model(model_name: str, weights, half=False, backend='triton'):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model name: {model_name}")

    model_config = MODEL_CONFIGS[model_name]

    if model_name in MODEL_CONFIGS:
        from mpttune.model.mpt.model import load_model
        model, tokenizer = load_model(model_config, weights, half=half, backend=backend)

    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model.eval()
    return model, tokenizer
