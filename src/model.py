import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from world import TAGS, CH_NAMES, N_OUT

BASE = 'Qwen/Qwen3-0.6B'


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

    def hidden(self, ids, chans, attn, pos=None, past=None, use_cache=False):
        x = self.base.get_input_embeddings()(ids)
        if self.use_chan:
            x = x + self.chan_emb(chans).to(x.dtype)
        out = self.base.model(inputs_embeds=x, attention_mask=attn, position_ids=pos,
                              past_key_values=past, use_cache=use_cache)
        return out.last_hidden_state, out.past_key_values

    def heads(self, h):
        return self.base.lm_head(h), (self.router(h) if self.use_router else None)
