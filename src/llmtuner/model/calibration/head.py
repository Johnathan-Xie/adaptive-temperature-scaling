import collections
import inspect
import os
import warnings
import math

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union, Tuple

import torch
from torch import nn

from transformers import AutoModelForCausalLM
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory, is_npu_available, is_xpu_available
from safetensors.torch import load_file as safe_load_file

from huggingface_hub import hf_hub_download, file_exists
from huggingface_hub.utils import EntryNotFoundError

from safetensors.torch import save_file as safe_save_file
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.utils import PushToHubMixin
from transformers.activations import ACT2FN
from peft.utils import id_tensor_storage

from .modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

class TemperatureHead(nn.Module):
    def __init__(self, init_temperature=0.0, **kwargs):
        super().__init__()
        self.temperature = nn.Parameter(torch.Tensor([init_temperature]))
    
    def forward(self, base_model_outputs, **kwargs):
        assert hasattr(base_model_outputs, "logits"), "Base Model must output logits for TemperatureHead"
        base_model_outputs.calibrated_logits = base_model_outputs.logits / self.temperature
        return base_model_outputs

class ElementWisePlattScalingHead(nn.Module):
    def __init__(self, init_temperature=0.0, in_features=32000, **kwargs):
        super().__init__()
        self.temperature = nn.Parameter(torch.Tensor([init_temperature] * in_features).unsqueeze(0).unsqueeze(0))
        self.bias = nn.Parameter(torch.Tensor([0] * in_features).unsqueeze(0).unsqueeze(0))
    
    def forward(self, base_model_outputs, **kwargs):
        assert hasattr(base_model_outputs, "logits"), "Base Model must output logits for TemperatureHead"
        base_model_outputs.calibrated_logits = base_model_outputs.logits * self.temperature + self.bias
        return base_model_outputs

class MatrixPlattScalingHead(nn.Module):
    def __init__(self, init_temperature=0.0, in_features=32000, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features)
    
    def forward(self, base_model_outputs, **kwargs):
        assert hasattr(base_model_outputs, "logits"), "Base Model must output logits for TemperatureHead"
        base_model_outputs.calibrated_logits = self.linear(base_model_outputs.logits)
        return base_model_outputs

def get_feature(feature_key, model_outputs, lm_head_weights=None):
    if feature_key == "hidden_states":
        features = model_outputs["hidden_states"][-1]
    elif feature_key == "maxlogit":
        features = model_outputs["logits"].max(dim=-1, keepdim=True)[0]
    elif feature_key == "output_token_feature":
        assert lm_head_weights is not None
        indices = model_outputs["logits"].argmax(dim=-1)
        features = lm_head_weights[indices]
    elif feature_key == "logits_std":
        features = torch.log(model_outputs["logits"].std(dim=-1, keepdim=True))
    else:
        features = model_outputs[feature_key]
    return features

class BaseAdaptiveCalibrationHead(nn.Module):
    def __init__(
        self,
        feature_key="hidden_states",
        prediction_type="temperature",
        max_temperature=10,
        base_model=None,
        head_module_name="linear",
        normalize_logits=False,
        **kwargs
    ):
        super().__init__()
        self.feature_key = feature_key
        self.max_temperature = max_temperature
        self.lm_head_weights = base_model.lm_head.weight.detach()
        self.prediction_type = prediction_type
        self.normalize_logits = normalize_logits

    def get_head_output(self, features, attention_mask=None):
        raise NotImplementedError
        
    def construct_features(self, base_model_outputs):
        features = []
        for feature_key in self.feature_key.split("+"):
            features.append(get_feature(feature_key, base_model_outputs, lm_head_weights=self.lm_head_weights))
        return torch.cat(features, dim=-1)
    
    def forward(self, base_model_outputs, attention_mask=None):
        features = self.construct_features(base_model_outputs)
        # Predicting a sigmoid is somewhat problematic because we can only really probe the confidence on the highest confidence token
        head_output = self.get_head_output(features, attention_mask)
        logits = base_model_outputs.logits
        if self.normalize_logits:
            logits = logits / logits.std(dim=-1, keepdim=True)
        if self.prediction_type == "temperature":
            base_model_outputs.calibrated_logits = logits * torch.exp(head_output).clip(max=self.max_temperature)
        elif self.prediction_type == "correctness":
            base_model_outputs.confidences = torch.sigmoid(head_output)
        return base_model_outputs

class LinearHead(BaseAdaptiveCalibrationHead):
    def __init__(self, in_features, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(in_features, 1)
    
    def get_head_output(self, features, *args, **kwargs):
        features = features.to(self.linear.weight.dtype)
        return self.linear(features)

class MLP(nn.Module):
    """https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L239C27-L240C1"""
    def __init__(self, in_features, intermediate_size, hidden_act="silu"):
        super().__init__()
        self.hidden_size = in_features
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class MLPHead(BaseAdaptiveCalibrationHead):
    def __init__(self, in_features, intermediate_size, **kwargs):
        super().__init__(**kwargs)
        self.mlp = MLP(in_features, intermediate_size)
        self.linear_head = nn.Linear(in_features, 1)

    def get_head_output(self, features, *args, **kwargs):
        # Residual connection
        features = features.to(self.linear_head.weight.dtype) 
        features = self.mlp(features)# + features
        return self.linear_head(features)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class SelfAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper
    https://github.com/huggingface/transformers/blob/bc72b4e2cdcbc80d5f56731f35dbc9c18b4c8de6/src/transformers/models/llama/modeling_llama.py#L285C1-L285C1
    Maybe modify later to support non-causal attention (We just need to adjust the attention mask to unmask any positions where they can already attend to themselves
    We still need to mask padding tokens tho.)
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout,
        num_key_value_heads,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        is_causal=True,
        layer_idx: Optional[int] = None,
        attention_bias=False,
        **kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.attention_dropout = attention_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.is_causal = is_causal

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class DecoderLayer(nn.Module):
    def __init__(
        self, 
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        attention_dropout=0.0,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        is_causal=True,
        layer_idx=None,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.self_attn = SelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            is_causal=is_causal,
            layer_idx=layer_idx
        )
        
        self.mlp = MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class TransformerHead(BaseAdaptiveCalibrationHead):
    def __init__(
        self, 
        in_features,
        intermediate_size,
        num_attention_heads,
        attention_dropout,
        num_key_value_heads,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        is_causal=True,
        layer_idx: Optional[int] = None,
        rms_norm_eps=1e-6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.transformer = DecoderLayer(
            hidden_size=in_features,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            is_causal=is_causal,
            layer_idx=layer_idx,
        )
        self.linear_head = nn.Linear(in_features, 1)

    def get_head_output(self, features, attention_mask, *args, **kwargs):
        # Residual connection
        features = features.to(self.linear_head.weight.dtype) 
        batch_size, seq_length = attention_mask.shape
        attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), features, 0
            )
        layer_outputs = self.transformer(features, attention_mask=attention_mask)

        return self.linear_head(layer_outputs[0])


ARCHITECTURE_TYPE_TO_HEAD_MAPPING = {
    "temperature": TemperatureHead,
    "linear": LinearHead,
    "platt_scaling_elementwise": ElementWisePlattScalingHead,
    "platt_scaling_matrix": MatrixPlattScalingHead,
    "mlp": MLPHead,
    "transformer": TransformerHead,
}