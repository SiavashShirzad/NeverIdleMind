"""Synthetic visual scenes + frozen CLIP-ConvNeXt 'visual cortex' features (16 tokens per image)."""
import random
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

COLORS = {'red': (220, 40, 40), 'green': (40, 170, 60), 'blue': (40, 80, 220), 'yellow': (230, 200, 30)}
SHAPES = ['circle', 'square', 'triangle']
N_VIS = 16


def make_scene(rng, size=256):
    img = Image.new('RGB', (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    objs, boxes = [], []
    for _ in range(rng.randint(1, 6)):
        for _try in range(50):
            r = rng.randint(16, 28)
            x, y = rng.randint(r + 4, size - r - 4), rng.randint(r + 4, size - r - 4)
            if all(abs(x - bx) > r + br + 4 or abs(y - by) > r + br + 4 for bx, by, br in boxes):
                break
        else:
            continue
        c, s = rng.choice(list(COLORS)), rng.choice(SHAPES)
        if s == 'circle':
            d.ellipse([x - r, y - r, x + r, y + r], fill=COLORS[c])
        elif s == 'square':
            d.rectangle([x - r, y - r, x + r, y + r], fill=COLORS[c])
        else:
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=COLORS[c])
        boxes.append((x, y, r)); objs.append((c, s))
    return img, objs


def question(rng, objs):
    kinds = ['count_color', 'count_shape', 'exists']
    if any(sum(1 for _, s in objs if s == sh) == 1 for sh in SHAPES):
        kinds.append('color_of')
    k = rng.choice(kinds)
    if k == 'count_color':
        c = rng.choice(list(COLORS))
        return k, f"how many {c} shapes?", str(sum(1 for cc, _ in objs if cc == c))
    if k == 'count_shape':
        s = rng.choice(SHAPES)
        return k, f"how many {s}s?", str(sum(1 for _, ss in objs if ss == s))
    if k == 'exists':
        c, s = rng.choice(list(COLORS)), rng.choice(SHAPES)
        return k, f"is there a {c} {s}?", 'yes' if (c, s) in objs else 'no'
    s = rng.choice([sh for sh in SHAPES if sum(1 for _, x in objs if x == sh) == 1])
    return k, f"what color is the {s}?", next(c for c, x in objs if x == s)


@torch.no_grad()
def encode(n, seed, out, bs=128):
    import timm
    m = timm.create_model('convnext_base.clip_laion2b_augreg_ft_in12k', pretrained=True).cuda().eval().half()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda')[:, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda')[:, None, None]
    rng = random.Random(seed)
    feats, metas = [], []
    for i in range(0, n, bs):
        imgs = []
        for _ in range(min(bs, n - i)):
            img, objs = make_scene(rng)
            imgs.append(torch.from_numpy(np.asarray(img)).permute(2, 0, 1)); metas.append(objs)
        x = ((torch.stack(imgs).cuda().float() / 255 - mean) / std).half()
        f = m.forward_features(x)                       # B,1024,8,8
        f = torch.nn.functional.adaptive_avg_pool2d(f, 4).flatten(2).transpose(1, 2)  # B,16,1024
        feats.append(f.cpu())
    torch.save(dict(feats=torch.cat(feats), objs=metas), out)


if __name__ == '__main__':
    encode(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
