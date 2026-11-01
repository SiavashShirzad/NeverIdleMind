import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from world import TAGS, CH_NAMES, N_OUT, N_VIS, N_AUD

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
            for tag in TAGS:
                ids = tk(' ' + tag[2:-2], add_special_tokens=False).input_ids
                emb[tk.convert_tokens_to_ids(tag)] = emb[ids].mean(0)
        self.chan_emb = nn.Embedding(len(CH_NAMES), d)
        nn.init.zeros_(self.chan_emb.weight)
        self.router = nn.Linear(d, N_OUT)
        self.vis_proj = nn.Sequential(nn.Linear(1024, d), nn.GELU(), nn.Linear(d, d))  # frozen-encoder features -> workspace
        self.vis_pos = nn.Embedding(N_VIS, d)
        nn.init.zeros_(self.vis_pos.weight)
        self.register_buffer('vis_bank', torch.zeros(1, 1024), persistent=False)
        self.aud_proj = nn.Sequential(nn.Linear(512, d), nn.GELU(), nn.Linear(d, d))  # frozen Whisper encoder -> workspace
        self.aud_pos = nn.Embedding(N_AUD, d)
        nn.init.zeros_(self.aud_pos.weight)
        self.register_buffer('aud_bank', torch.zeros(1, 512), persistent=False)

    def hidden(self, ids, chans, attn, pos=None, past=None, use_cache=False, img=None):
        x = self.base.get_input_embeddings()(ids)
        if img is not None and bool((img >= 0).any()):
            sel = img >= 0
            f = self.vis_proj(self.vis_bank[img[sel]].to(self.vis_proj[0].weight.dtype)) + self.vis_pos(img[sel] % N_VIS)
            x = x.clone(); x[sel] = f.to(x.dtype)
        if img is not None and bool((img <= -2).any()):
            sel = img <= -2
            row = -img[sel] - 2
            f = self.aud_proj(self.aud_bank[row].to(self.aud_proj[0].weight.dtype)) + self.aud_pos(row % N_AUD)
            x = x.clone(); x[sel] = f.to(x.dtype)
        if self.use_chan:
            x = x + self.chan_emb(chans).to(x.dtype)
        out = self.base.model(inputs_embeds=x, attention_mask=attn, position_ids=pos,
                              past_key_values=past, use_cache=use_cache)
        return out.last_hidden_state, out.past_key_values

    def heads(self, h):
        return self.base.lm_head(h), (self.router(h) if self.use_router else None)
