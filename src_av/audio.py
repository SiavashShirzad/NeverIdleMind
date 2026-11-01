"""Heard utterances: two spoken digits (Free Spoken Digit Dataset) -> frozen Whisper-base encoder 'auditory cortex',
pooled to 16 tokens per utterance. Speaker 'theo' is held out for test."""
import glob
import json
import os
import random
import sys

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

REC = 'data/ext/fsdd/recordings'
TEST_SPK = 'theo'
N_AUD = 16


def clips(test):
    out = {}
    for f in sorted(glob.glob(f'{REC}/*.wav')):
        d, spk, _ = os.path.basename(f)[:-4].split('_')
        if (spk == TEST_SPK) == test:
            out.setdefault(spk, []).append((int(d), f))
    return out


def load(f):
    sr, x = wavfile.read(f)
    x = x.astype(np.float32) / 32768.0
    return resample_poly(x, 2, 1) if sr == 8000 else x  # -> 16 kHz


@torch.no_grad()
def encode(n, seed, test, out, bs=64):
    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-base')
    enc = WhisperModel.from_pretrained('openai/whisper-base').encoder.cuda().eval().half()
    rng, pool = random.Random(seed), clips(test)
    spks, feats, meta = sorted(pool), [], []
    for i in range(0, n, bs):
        wav = []
        for _ in range(min(bs, n - i)):
            spk = rng.choice(spks)
            (a, fa), (b, fb) = rng.choice(pool[spk]), rng.choice(pool[spk])
            gap = np.zeros(int(16000 * rng.uniform(0.15, 0.4)), np.float32)
            wav.append(np.concatenate([load(fa), gap, load(fb)])); meta.append([a, b])
        x = fe(wav, sampling_rate=16000, return_tensors='pt').input_features.cuda().half()
        h = enc(x).last_hidden_state[:, :150]                     # first 3 s (50 frames/s)
        h = torch.nn.functional.adaptive_avg_pool1d(h.transpose(1, 2).float(), N_AUD).transpose(1, 2)  # B,16,512
        feats.append(h.half().cpu())
    os.makedirs('data/aud', exist_ok=True)
    torch.save(dict(feats=torch.cat(feats), meta=meta), out)
    json.dump(meta, open(out.replace('.pt', '_meta.json'), 'w'))


if __name__ == '__main__':
    encode(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == 'test', sys.argv[4])
