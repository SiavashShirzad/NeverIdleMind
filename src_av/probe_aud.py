"""Linear probes on frozen Whisper features: first / second spoken digit; test speaker held out."""
import torch

tr, te = torch.load('data/aud/train.pt'), torch.load('data/aud/test.pt')
Xtr, Xte = tr['feats'].float().flatten(1), te['feats'].float().flatten(1)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
Xtr, Xte = ((Xtr - mu) / sd).cuda(), ((Xte - mu) / sd).cuda()
Ytr, Yte = torch.tensor(tr['meta']), torch.tensor(te['meta'])
for k, name in enumerate(['first digit', 'second digit']):
    W = torch.nn.Linear(Xtr.shape[1], 10).cuda(); opt = torch.optim.AdamW(W.parameters(), 1e-3, weight_decay=0.05)
    for _ in range(300):
        opt.zero_grad(); torch.nn.functional.cross_entropy(W(Xtr), Ytr[:, k].cuda()).backward(); opt.step()
    print(f'{name}: probe {(W(Xte).argmax(1).cpu() == Yte[:, k]).float().mean().item():.3f} (chance 0.100)')
