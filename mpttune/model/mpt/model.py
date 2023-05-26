import math
import warnings
import importlib
from typing import Dict, Optional, Union, List, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import transformers
import accelerate
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from einops import rearrange

from mpttune.backend.base import replace_4bit_linear, find_layers

logger = logging.get_logger("transformers")


class MPTConfig(PretrainedConfig):
    model_type = 'mpt'

    def __init__(self,
                 d_model: int = 4096,
                 n_heads: int = 32,
                 n_layers: int = 32,
                 expansion_ratio: int = 4,
                 max_seq_len: int = 2048,
                 vocab_size: int = 50432,
                 resid_pdrop: float = 0.0,
                 emb_pdrop: float = 0.0,
                 learned_pos_emb: bool = True,
                 initializer_range=0.02,

                 architectures: List[str] = [
                     "MPTForCausalLM"
                 ],

                 attn_config: Dict = {
                     'attn_type': 'multihead_attention',
                     'attn_pdrop': 0.0,
                     'attn_impl': 'torch',
                     'qk_ln': False,
                     'clip_qkv': False,
                     'softmax_scale': None,
                     'prefix_lm': False,
                     'attn_uses_sequence_id': False,
                     'alibi': True,
                     'alibi_bias_max': 8},

                 init_device: str = 'cpu',
                 logit_scale: Optional[Union[float, str]] = None,
                 no_bias: bool = True,
                 verbose: int = 0,
                 embedding_fraction: float = 1.0,
                 norm_type: str = 'low_precision_layernorm',
                 use_cache: bool = False,
                 tokenizer_name: str = "EleutherAI/gpt-neox-20b",
                 torch_dtype: str = "bfloat16",
                 transformers_version: str = "4.29.2",

                 init_config: Dict = {
                     "emb_init_std": None,
                     "emb_init_uniform_lim": None,
                     "fan_mode": "fan_in",
                     "init_div_is_residual": True,
                     "init_gain": 0,
                     "init_nonlinearity": "relu",
                     "init_std": 0.02,
                     "name": "kaiming_normal_",
                     "verbose": 0
                 },

                 **kwargs):
        """The MPT configuration class.
        Args:
            d_model (int): The size of the embedding dimension of the model.
            n_heads (int): The number of attention heads.
            n_layers (int): The number of layers in the model.
            expansion_ratio (int): The ratio of the up/down scale in the MLP.
            max_seq_len (int): The maximum sequence length of the model.
            vocab_size (int): The size of the vocabulary.
            resid_pdrop (float): The dropout probability applied to the attention output before combining with residual.
            emb_pdrop (float): The dropout probability for the embedding layer.
            learned_pos_emb (bool): Whether to use learned positional embeddings
            attn_config (Dict):  A dictionary used to configure the model's attention module:
                attn_type (str): type of attention to use. Options: multihead_attention, multiquery_attention
                attn_pdrop (float): The dropout probability for the attention layers.
                attn_impl (str): The attention implementation to use. One of 'torch', 'flash', or 'triton'.
                qk_ln (bool): Whether to apply layer normalization to the queries and keys in the attention layer.
                clip_qkv (Optional[float]): If not None, clip the queries, keys, and values in the attention layer to
                    this value.
                softmax_scale (Optional[float]): If not None, scale the softmax in the attention layer by this value. If None,
                    use the default scale of ``1/sqrt(d_keys)``.
                prefix_lm (Optional[bool]): Whether the model should operate as a Prefix LM. This requires passing an
                    extra `prefix_mask` argument which indicates which tokens belong to the prefix. Tokens in the prefix
                    can attend to one another bi-directionally. Tokens outside the prefix use causal attention.
                attn_uses_sequence_id (Optional[bool]): Whether to restrict attention to tokens that have the same sequence_id.
                    When the model is in `train` mode, this requires passing an extra `sequence_id` argument which indicates
                    which sub-sequence each token belongs to.
                    Defaults to ``False`` meaning any provided `sequence_id` will be ignored.
                alibi (bool): Whether to use the alibi bias instead of position embeddings.
                alibi_bias_max (int): The maximum value of the alibi bias.
            init_device (str): The device to use for parameter initialization.
            logit_scale (Optional[Union[float, str]]): If not None, scale the logits by this value.
            no_bias (bool): Whether to use bias in all layers.
            verbose (int): The verbosity level. 0 is silent.
            embedding_fraction (float): The fraction to scale the gradients of the embedding layer by.
            norm_type (str): choose type of norm to use
            multiquery_attention (bool): Whether to use multiquery attention implementation.
            use_cache (bool): Whether or not the model should return the last key/values attentions
            init_config (Dict): A dictionary used to configure the model initialization:
                init_config.name: The parameter initialization scheme to use. Options: 'default_', 'baseline_',
                    'kaiming_uniform_', 'kaiming_normal_', 'neox_init_', 'small_init_', 'xavier_uniform_', or
                    'xavier_normal_'. These mimic the parameter initialization methods in PyTorch.
                init_div_is_residual (Union[int, float, str, bool]): Value to divide initial weights by if ``module._is_residual`` is True.
                emb_init_std (Optional[float]): The standard deviation of the normal distribution used to initialize the embedding layer.
                emb_init_uniform_lim (Optional[Union[Tuple[float, float], float]]): The lower and upper limits of the uniform distribution
                    used to initialize the embedding layer. Mutually exclusive with ``emb_init_std``.
                init_std (float): The standard deviation of the normal distribution used to initialize the model,
                    if using the baseline_ parameter initialization scheme.
                init_gain (float): The gain to use for parameter initialization with kaiming or xavier initialization schemes.
                fan_mode (str): The fan mode to use for parameter initialization with kaiming initialization schemes.
                init_nonlinearity (str): The nonlinearity to use for parameter initialization with kaiming initialization schemes.
                ---
                See llmfoundry.model.utils.param_init_fns.py for info on other param init config options
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.attn_config = attn_config
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.use_cache = use_cache
        self.init_config = init_config
        self.initializer_range = initializer_range
        self.tokenizer_name = tokenizer_name
        self.torch_dtype = torch_dtype
        self.transformers_version = transformers_version

        super().__init__(**kwargs)


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor


class LPLayerNorm(torch.nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine,
                         device=device, dtype=dtype)

    def forward(self, x):
        module_device = x.device

        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias

        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias,
                                                  self.eps)


def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f'tensor.dtype={tensor.dtype!r} must be in valid_dtypes={valid_dtypes!r}.')
        if not tensor.is_cuda:
            raise TypeError(f'Inputs must be cuda tensors (tensor.is_cuda={tensor.is_cuda!r}).')


def _reset_is_causal(num_query_tokens: int, num_key_tokens: int, original_is_causal: bool):
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError('MPT does not support query and key with different number of tokens, unless number of query tokens is 1.')
        else:
            return False
    return original_is_causal


def scaled_multihead_dot_product_attention(
        query, key, value,
        n_heads,
        softmax_scale=None,
        attn_bias=None,
        key_padding_mask=None,
        is_causal=False,
        dropout_p=0.0,
        training=False,
        needs_weights=False,
        multiquery=False):
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    k = rearrange(key, 'b s (h d) -> b h d s', h=1 if multiquery else n_heads)
    v = rearrange(value, 'b s (h d) -> b h s d', h=1 if multiquery else n_heads)

    min_val = torch.finfo(q.dtype).min

    (b, _, s_q, d) = q.shape

    s_k = k.size(-1)

    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)

    attn_weight = q.matmul(k) * softmax_scale

    if attn_bias is not None:
        if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k or (
                attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q):
            raise RuntimeError(
                f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.')
        attn_weight = attn_weight + attn_bias

    if key_padding_mask is not None:
        raise NotImplementedError

    if is_causal:
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)

    # upcast attention to fp32
    attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)

    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=training, inplace=True)

    out = attn_weight.matmul(v)

    out = rearrange(out, 'b h s d -> b s (h d)')

    if needs_weights:
        return (out, attn_weight)

    return (out, None)


def triton_flash_attn_fn(
        query, key, value,
        n_heads,
        softmax_scale=None,
        attn_bias=None,
        key_padding_mask=None,
        is_causal=False,
        dropout_p=0.0,
        training=False,
        needs_weights=False,
        multiquery=False):
    from mpttune.backend.triton import flash_attn_triton

    check_valid_inputs(query, key, value)

    if dropout_p:
        raise NotImplementedError(f'Dropout not implemented for attn_impl: triton.')

    if needs_weights:
        raise NotImplementedError(f'attn_impl: triton cannot return attn weights.')

    if key_padding_mask is not None:
        warnings.warn('Propagating key_padding_mask to the attention module ' +
                      'and applying it within the attention module can cause ' +
                      'unnecessary computation/memory usage. Consider integrating ' +
                      'into attn_bias once and passing that to each attention ' +
                      'module instead.')
        (b_size, s_k) = key_padding_mask.shape[:2]

        if attn_bias is None:
            attn_bias = query.new_zeros(b_size, 1, 1, s_k)

        attn_bias = attn_bias.masked_fill(~key_padding_mask.view((b_size, 1, 1, s_k)), torch.finfo(query.dtype).min)

    query = rearrange(query, 'b s (h d) -> b s h d', h=n_heads)
    key = rearrange(key, 'b s (h d) -> b s h d', h=1 if multiquery else n_heads)
    value = rearrange(value, 'b s (h d) -> b s h d', h=1 if multiquery else n_heads)

    if multiquery:
        key = key.expand(*key.shape[:2], n_heads, key.size(-1))
        value = value.expand(*value.shape[:2], n_heads, value.size(-1))

    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)

    attn_output = flash_attn_triton.flash_attn_func(
        query, key, value,
        attn_bias, reset_is_causal, softmax_scale)

    output = attn_output.view(*attn_output.shape[:2], -1)
    return (output, None)


class MPTAttention(nn.Module):
    """Multi-head self attention.
    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(self, config: MPTConfig):
        super().__init__()

        self.attn_impl = config.attn_config['attn_impl']
        self.clip_qkv = config.attn_config['clip_qkv']
        self.qk_ln = config.attn_config['qk_ln']

        if self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
        else:
            raise NotImplemented("supported attention: ['torch']")

        logger.debug(f'using {self.attn_fn} attention function')

        self.config = config
        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads

        assert (self.head_dim * self.num_heads) == self.hidden_size

        self.max_position_embeddings = config.max_seq_len

        self.softmax_scale = config.attn_config['softmax_scale']
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.hidden_size / self.num_heads)

        self.attn_dropout_p = config.attn_config['attn_pdrop']

        self.Wqkv = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)

        if self.qk_ln:
            assert config['norm_type'] == 'low_precision_layernorm'
            self.q_ln = LPLayerNorm(self.hidden_size)
            self.k_ln = LPLayerNorm(self.hidden_size)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            attn_bias: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        qkv = self.Wqkv(hidden_states)

        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)

        (query, key, value) = qkv.chunk(3, dim=2)

        key_padding_mask = attention_mask

        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)

        if past_key_value is not None:
            if len(past_key_value) != 0:
                key = torch.cat([past_key_value[0], key], dim=1)
                value = torch.cat([past_key_value[1], value], dim=1)

            past_key_value = (key, value) if use_cache else None

        if attn_bias is not None:
            attn_bias = attn_bias[:, :, -query.size(1):, -key.size(1):]

        (context, attn_weights) = self.attn_fn(
            query, key, value,
            self.num_heads,
            softmax_scale=self.softmax_scale,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            is_causal=True,
            dropout_p=self.attn_dropout_p,
            training=self.training,
            needs_weights=output_attentions)

        return (self.out_proj(context), attn_weights, past_key_value)


class MPTBlock(nn.Module):
    def __init__(self, config: MPTConfig):
        super().__init__()

        assert config.attn_config['attn_type'] == 'multihead_attention'
        assert config.norm_type == 'low_precision_layernorm'

        self.norm_1 = LPLayerNorm(config.d_model)

        self.attn = MPTAttention(config)

        self.norm_2 = LPLayerNorm(config.d_model)

        self.ffn = MPTMLP(
            hidden_size=config.d_model,
            intermediate_size=config.expansion_ratio * config.d_model)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            attn_bias: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        a = self.norm_1(hidden_states)

        (b, self_attn_weights, present_key_value) = self.attn(
            hidden_states=a,
            attention_mask=attention_mask,
            attn_bias=attn_bias,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache)

        hidden_states = hidden_states + self.resid_attn_dropout(b)
        m = self.norm_2(hidden_states)
        n = self.ffn(m)
        hidden_states = hidden_states + self.resid_ffn_dropout(n)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MPTMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.act_fn = nn.GELU(approximate='none')
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class MPTPreTrainedModel(PreTrainedModel):
    config_class = MPTConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ["MPTBlock"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MPTModel):
            module.gradient_checkpointing = value


def build_attn_bias(attn_impl, attn_bias, n_heads, seq_len, causal=False, alibi=False, alibi_bias_max=8):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            (device, dtype) = (attn_bias.device, attn_bias.dtype)
            attn_bias = attn_bias.add(
                build_alibi_bias(n_heads, seq_len, full=not causal, alibi_bias_max=alibi_bias_max, device=device,
                                 dtype=dtype))
        return attn_bias
    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')


def gen_slopes(n_heads, alibi_bias_max=8, device=None):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = 1.0 / torch.pow(2, m)
    if _n_heads != n_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes.view(1, n_heads, 1, 1)


def build_alibi_bias(n_heads, seq_len, full=False, alibi_bias_max=8, device=None, dtype=None):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, 1, seq_len)

    if full:
        alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)

    slopes = gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)


def attn_bias_shape(attn_impl, n_heads, seq_len, alibi, prefix_lm, causal, use_sequence_id):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            if (prefix_lm or not causal) or use_sequence_id:
                return (1, n_heads, seq_len, seq_len)
            return (1, n_heads, 1, seq_len)
        elif prefix_lm or use_sequence_id:
            return (1, 1, seq_len, seq_len)
        return None
    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')


class MPTModel(MPTPreTrainedModel):
    def __init__(self, config: MPTConfig):
        super().__init__(config)
        assert config.no_bias

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False

        self.attn_impl = config.attn_config['attn_impl']
        self.prefix_lm = config.attn_config['prefix_lm']
        self.attn_uses_sequence_id = config.attn_config['attn_uses_sequence_id']
        self.alibi = config.attn_config['alibi']
        self.alibi_bias_max = config.attn_config['alibi_bias_max']
        self.embedding_fraction = config.embedding_fraction

        self.wte = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.emb_drop = nn.Dropout(config.emb_pdrop)
        self.blocks = nn.ModuleList([MPTBlock(config) for _ in range(config.n_layers)])
        self.norm_f = LPLayerNorm(config.d_model)

        self.is_causal = True
        self.attn_bias = None

        self.attn_bias_shape = attn_bias_shape(
            self.attn_impl,
            config.n_heads,
            config.max_seq_len,
            self.alibi,
            prefix_lm=self.prefix_lm,
            causal=self.is_causal,
            use_sequence_id=self.attn_uses_sequence_id)

        self._attn_bias_initialized = False

        # Initialize weights and apply final processing
        self.post_init()

        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                    logger.debug(f'Removing bias ({module.bias}) from {module}.')
                    module.register_parameter('bias', None)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, value):
        self.wte = value

    @torch.no_grad()
    def _attn_bias(
            self,
            device,
            dtype,
            attention_mask: Optional[torch.ByteTensor] = None,
            prefix_mask: Optional[torch.ByteTensor] = None,
            sequence_id: Optional[torch.LongTensor] = None):

        if not self._attn_bias_initialized:
            if self.attn_bias_shape:
                self.attn_bias = torch.zeros(self.attn_bias_shape, device=device, dtype=dtype)

                self.attn_bias = build_attn_bias(
                    self.attn_impl,
                    self.attn_bias,
                    self.config.n_heads,
                    self.config.max_seq_len,
                    causal=True,
                    alibi=self.alibi,
                    alibi_bias_max=self.alibi_bias_max)

            self._attn_bias_initialized = True

        if self.attn_impl == 'flash':
            raise NotImplementedError

        if self.attn_bias is not None:
            self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)

        attn_bias = self.attn_bias

        if self.prefix_lm:
            raise NotImplementedError

        if self.attn_uses_sequence_id and sequence_id is not None:
            raise NotImplementedError

        if attention_mask is not None:
            s_k = attention_mask.shape[-1]

            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k), device=device, dtype=dtype)
            else:
                attn_bias = attn_bias[:, :, :, -s_k:]

            if prefix_mask is not None and attention_mask.shape != prefix_mask.shape:
                raise ValueError(
                    f'attention_mask shape={attention_mask.shape} ' + f'and prefix_mask shape={prefix_mask.shape} are not equal.')

            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(~attention_mask.view(-1, 1, 1, s_k), min_val)

        return (attn_bias, None)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            prefix_mask: Optional[torch.ByteTensor] = None,
            sequence_id: Optional[torch.LongTensor] = None) -> Union[Tuple, BaseModelOutputWithPast]:

        assert position_ids is None
        assert sequence_id is None
        assert prefix_mask is None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        assert seq_length <= self.config.max_seq_len, f'Cannot forward input with seq_len=X, this model only supports seq_len<={self.config.max_seq_len}'

        if attention_mask is not None:
            attention_mask = attention_mask.bool()

        if prefix_mask is not None:
            prefix_mask = prefix_mask.bool()

        if not return_dict:
            raise NotImplementedError('return_dict False is not implemented yet for MPT')

        if attention_mask is not None and attention_mask[:, 0].sum() != attention_mask.shape[0] and self.training:
            raise NotImplementedError('MPT does not support training with left padding.')

        if self.prefix_lm and prefix_mask is None:
            raise ValueError('prefix_mask is a required argument when MPT is configured with prefix_lm=True.')

        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError(
                    'sequence_id is a required argument when MPT is configured with attn_uses_sequence_id=True ' + 'and the model is in train mode.')
            elif self.attn_uses_sequence_id is False and sequence_id is not None:
                raise NotImplementedError

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if self.alibi:
            pass
        else:
            raise NotImplementedError

        if self.embedding_fraction == 1:
            inputs_embeds = self.emb_drop(inputs_embeds)
        else:
            raise NotImplementedError

        (attn_bias, attention_mask) = self._attn_bias(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id)

        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)]

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for (idx, decoder_layer) in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    attn_bias,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    attn_bias=attn_bias,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm_f(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MPTForCausalLM(MPTPreTrainedModel):
    def __init__(self, config: MPTConfig):
        super().__init__(config)
        if not config.tie_word_embeddings:
            raise ValueError('MPTForCausalLM only supports tied word embeddings')

        self.transformer = MPTModel(config)

        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(
                        f"logit_scale={logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.")
            self.logit_scale = logit_scale

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.transformer.wte

    def set_output_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        hidden_states = outputs[0]

        logits = F.linear(hidden_states, self.transformer.wte.weight)
        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f'Multiplying logits by self.logit_scale={self.logit_scale!r}. This will produce uniform (uninformative) outputs.')
            logits *= self.logit_scale

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
                                      **kwargs):
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
        else:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=input_ids.device)

        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError('MPT does not support generation with right padding.')

        if self.transformer.attn_uses_sequence_id and self.training:
            raise NotImplementedError
        else:
            sequence_id = None

        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if self.transformer.prefix_lm:
            raise NotImplementedError
        else:
            prefix_mask = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": None,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
            }
        )

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


def load_model(llm_config, checkpoint, half=False, backend='triton'):
    config = MPTConfig.from_pretrained(llm_config.hf_config_name)
    config.max_seq_len = llm_config.max_seq_len

    assert config.attn_config['attn_impl'] in ['torch', 'triton']
    config.attn_config['attn_impl'] = llm_config.attn_impl

    assert config.attn_config['alibi'] is True

    if half:
        torch.set_default_dtype(torch.half)

    if (llm_config.bits == 4) and llm_config.groupsize:
        with accelerate.init_empty_weights():
            ql = importlib.import_module(f'mpttune.backend.{backend}.quantlinear')

            model = MPTForCausalLM(config)
            model = model.eval()

            replace_4bit_linear(
                model,
                find_layers(model),
                llm_config.bits,
                llm_config.groupsize,
                quantlinear_class=ql.QuantLinear
            )

        model = accelerate.load_checkpoint_and_dispatch(
            model=model, checkpoint=checkpoint, device_map=llm_config.device_map,
            no_split_module_classes=["MPTBlock"]
        )

        model.loaded_in_4bit = True

    elif llm_config.bits == 8:
        model = MPTForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            load_in_8bit=True,
            device_map=llm_config.device_map
        )
        model.loaded_in_8bit = True

    else:
        model = MPTForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=llm_config.device_map
        )
        model.loaded_in_bf16 = True

    if config.no_bias:
        for module in model.modules():
            if hasattr(module, 'bias'):
                logger.debug(f'Removing bias ({module.bias}) from {module}.')
                module.bias = None

    model.seqlen = llm_config.max_seq_len

    tokenizer = transformers.AutoTokenizer.from_pretrained(llm_config.hf_tokenizer_config)
    if tokenizer.pad_token is None:
        tokenizer.add_tokens('<pad>', special_tokens=True)
        tokenizer.pad_token = '<pad>'
        assert tokenizer.pad_token_id is not None

    return model, tokenizer
