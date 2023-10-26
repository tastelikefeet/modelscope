# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import warnings
from typing import Optional, Tuple

import torch
from transformers.models.llama import LlamaConfig
from transformers.models.llama import LlamaModel as LlamaModelHF
from transformers.models.llama import \
    LlamaPreTrainedModel as LlamaPreTrainedModelHF
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      apply_rotary_pos_emb)

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


class MsModelMixin:

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.

        Args:
            kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the label information.
                    num_labels: An optional arg to tell the model how many classes to initialize.
                                    Method will call utils.parse_label_mapping if num_labels not supplied.
                                    If num_labels is not found, the model will use the default setting (2 classes).

        Returns:
            The loaded model, which is initialized by transformers.PreTrainedModel.from_pretrained
        """
        model_dir = kwargs.pop('model_dir', None)
        if model_dir is None:
            config = LlamaConfig(**kwargs)
            model = cls(config)
        else:
            model = super(MsModelMixin, cls).from_pretrained(
                pretrained_model_name_or_path=model_dir, **kwargs)
        model.model_dir = model_dir
        return model


class LlamaPreTrainedModel(MsModelMixin, LlamaPreTrainedModelHF, TorchModel):
    pass


class LlamaFlashAttention(LlamaAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
        from flash_attn.bert_padding import unpad_input, pad_input
        if output_attentions:
            warnings.warn(
                'Output attentions is not supported for patched `LlamaAttention`, returning `None` instead.'
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                            self.head_dim).transpose(1, 2))
        key_states = (
            self.k_proj(hidden_states).view(bsz, q_len, self.num_heads,
                                            self.head_dim).transpose(1, 2))
        value_states = (self.v_proj(hidden_states).view(
            bsz, q_len, self.num_heads,
            self.head_dim).transpose(1,
                                     2))  # shape: (b, num_heads, s, head_dim)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Transform the data into the format required by flash attention
        qkv = torch.stack([query_states, key_states, value_states], dim=2)
        qkv = qkv.transpose(1, 3)  # shape: [b, s, 3, num_heads, head_dim]
        key_padding_mask = attention_mask

        if key_padding_mask is None:
            qkv = qkv.reshape(-1, 3, self.num_heads, self.head_dim)
            cu_q_lens = torch.arange(
                0, (bsz + 1) * q_len,
                step=q_len,
                dtype=torch.int32,
                device=qkv.device)
            max_s = q_len
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True)
            output = output.view(bsz, q_len, -1)
        else:
            qkv = qkv.reshape(bsz, q_len, -1)
            qkv, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
            output_unpad = flash_attn_varlen_qkvpacked_func(
                qkv, cu_q_lens, max_s, 0.0, softmax_scale=None, causal=True)
            output_unpad = output_unpad.reshape(-1,
                                                self.num_heads * self.head_dim)
            output = pad_input(output_unpad, indices, bsz, q_len)

        return self.o_proj(output), None, past_key_value


@MODELS.register_module(Tasks.backbone, module_name=Models.llama2)
@MODELS.register_module(Tasks.backbone, module_name=Models.llama)
class LlamaModel(MsModelMixin, LlamaModelHF, TorchModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        if getattr(self.config, 'use_flash_attn', False):
            LlamaAttention.forward = LlamaFlashAttention.forward

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, past_key_values_length):
        if getattr(self.config, 'use_flash_attn', False):
            return attention_mask
        else:
            return super()._prepare_decoder_attention_mask(
                attention_mask, input_shape, inputs_embeds,
                past_key_values_length)
