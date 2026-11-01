"""Linear probes on frozen visual features: per-colour / per-shape object counts (train 20k images, test 1k)."""
import torch

C, S = ['red', 'green', 'blue', 'yellow'], ['circle', 'square', 'triangle']
tr, te = torch.load('data/vis/train.pt'), torch.load('data/vis/test.pt')
y = lambda objs: torch.tensor([[sum(1 for c, _ in o if c == cc) for cc in C] + [sum(1 for _, s in o if s == ss) for ss in S] for o in objs])
Xtr, Xte = tr['feats'].float().flatten(1), te['feats'].float().flatten(1)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
Xtr, Xte = ((Xtr - mu) / sd).cuda(), ((Xte - mu) / sd).cuda()
Ytr, Yte = y(tr['objs']), y(te['objs'])
for k, name in enumerate(C + S):
    W = torch.nn.Linear(Xtr.shape[1], 7).cuda(); opt = torch.optim.AdamW(W.parameters(), 1e-3, weight_decay=0.05)
    for _ in range(300):
        opt.zero_grad(); torch.nn.functional.cross_entropy(W(Xtr), Ytr[:, k].cuda()).backward(); opt.step()
    acc = (W(Xte).argmax(1).cpu() == Yte[:, k]).float().mean().item()
    maj = (Yte[:, k] == Ytr[:, k].mode().values).float().mean().item()
    print(f'{name}: probe {acc:.3f} majority {maj:.3f}')
