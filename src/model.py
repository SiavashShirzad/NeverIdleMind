import math
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from world import TAGS, CH_NAMES, N_OUT

BASE = os.environ.get('BASE_MODEL', 'Qwen/Qwen3-0.6B')
LORA_R = int(os.environ.get('LORA_R', 0))  # 0 = full fine-tuning
LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.scale


def load_tokenizer():
    tk = AutoTokenizer.from_pretrained(BASE)
    tk.add_special_tokens({'additional_special_tokens': TAGS})
    return tk


class ChannelLM(nn.Module):
    """Decoder + (optional) channel embedding on the input side and a router head on the output side.
    router=False gives the plain LM used by the tag/orchestrator/turn-based arms."""

    def __init__(self, router, tk, dtype=torch.float32, chan_emb=True):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype, attn_implementation='sdpa')
        self.use_router, self.use_chan = router, router and chan_emb
        d = self.base.config.hidden_size
        emb = self.base.get_input_embeddings().weight
        with torch.no_grad():  # initialise tag rows from the embedding of the channel's name
            for tag, name in zip(TAGS, CH_NAMES):
                ids = tk(' ' + name, add_special_tokens=False).input_ids
                emb[tk.convert_tokens_to_ids(tag)] = emb[ids].mean(0)
        self.chan_emb = nn.Embedding(len(CH_NAMES), d)
        nn.init.zeros_(self.chan_emb.weight)
        self.router = nn.Linear(d, N_OUT)
        self.lora = LORA_R > 0
        if self.lora:  # frozen base + adapters; tag rows get a trainable delta shared by input and (tied) output
            self.base.requires_grad_(False)
            for layer in self.base.model.layers:
                for parent in (layer.self_attn, layer.mlp):
                    for n in LORA_TARGETS:
                        if hasattr(parent, n):
                            setattr(parent, n, LoRALinear(getattr(parent, n), LORA_R, 2 * LORA_R))
            self.register_buffer('tag_ids', torch.tensor(tk.convert_tokens_to_ids(TAGS)), persistent=False)
            self.tag_delta = nn.Parameter(torch.zeros(len(TAGS), d))

    def hidden(self, ids, chans, attn, pos=None, past=None, use_cache=False):
        x = self.base.get_input_embeddings()(ids)
        if self.lora:
            hit = (ids[..., None] == self.tag_ids).to(x.dtype)
            x = x + hit @ self.tag_delta.to(x.dtype)
        if self.use_chan:
            x = x + self.chan_emb(chans).to(x.dtype)
        out = self.base.model(inputs_embeds=x, attention_mask=attn, position_ids=pos,
                              past_key_values=past, use_cache=use_cache)
        return out.last_hidden_state, out.past_key_values

    def logits(self, h):
        y = self.base.lm_head(h)
        if self.lora:
            y = y.index_add(-1, self.tag_ids, (h @ self.tag_delta.t().to(h.dtype)).to(y.dtype))
        return y

    def heads(self, h):
        return self.logits(h), (self.router(h) if self.use_router else None)

    def trainable_state(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.to(torch.bfloat16) for k, v in self.state_dict().items() if k in keep or not self.lora}
