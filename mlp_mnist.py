"""MNIST uzerinde sifirdan egitilen MLP (CNN yok, hazir egitilmis model yok).

Deney 1: Temel MLP  784 -> 256 -> 128 -> 10  (ReLU, CrossEntropyLoss, Adam)

Veri ayrimi:
  - MNIST train (60.000)  -> 50.000 train + 10.000 validation
  - MNIST test  (10.000)  -> yalnizca en sonda, tek seferde kullanilir

Kullanim:
  python mlp_mnist.py
  python mlp_mnist.py --hidden 512 256 --lr 0.0005 --epochs 30 --name deney_buyuk
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


class MLP(nn.Module):
    def __init__(self, hidden=(256, 128), dropout=0.0):
        super().__init__()
        layers, in_dim = [nn.Flatten()], 28 * 28
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 10))  # logit; softmax CrossEntropyLoss icinde
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_data(root, val_size, seed, device):
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)

    def prep(ds):
        x = ds.data.float().div(255.0)
        x = (x - MNIST_MEAN) / MNIST_STD  # normalizasyon
        return x.unsqueeze(1), ds.targets

    x_all, y_all = prep(tr)
    x_test, y_test = prep(te)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x_all), generator=g)
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    splits = {
        "train": (x_all[train_idx], y_all[train_idx]),
        "val": (x_all[val_idx], y_all[val_idx]),
        "test": (x_test, y_test),
    }
    return {k: (x.to(device), y.to(device)) for k, (x, y) in splits.items()}


@torch.no_grad()
def evaluate(model, x, y, criterion, bs=2000):
    model.eval()
    loss_sum, correct = 0.0, 0
    for i in range(0, len(x), bs):
        out = model(x[i:i + bs])
        loss_sum += criterion(out, y[i:i + bs]).item() * len(out)
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return loss_sum / len(x), correct / len(x)


def plot_history(h, path, title):
    ep = range(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, h["train_loss"], label="Train loss")
    ax[0].plot(ep, h["val_loss"], label="Validation loss")
    ax[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    ax[1].plot(ep, h["train_acc"], label="Train acc")
    ax[1].plot(ep, h["val_acc"], label="Validation acc")
    ax[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--val-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--name", default="deney1_temel_mlp")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_data(args.data_dir, args.val_size, args.seed, device)
    (xtr, ytr), (xva, yva), (xte, yte) = data["train"], data["val"], data["test"]
    print(f"Cihaz: {device} | train={len(xtr)} val={len(xva)} test={len(xte)}")

    model = MLP(args.hidden, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    hist = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(xtr), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = criterion(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        # epoch sonu: model.eval() ile tum train ve val uzerinde olcum
        tl, ta = evaluate(model, xtr, ytr, criterion)
        vl, va = evaluate(model, xva, yva, criterion)
        for k, v in zip(hist, [tl, vl, ta, va]):
            hist[k].append(v)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tl:.4f} acc {ta:.4f} "
              f"| val loss {vl:.4f} acc {va:.4f}")
    train_time = time.time() - start

    # Test seti yalnizca burada, egitim bittikten sonra bir kez kullanilir.
    test_loss, test_acc = evaluate(model, xte, yte, criterion)
    print(f"\nTest loss {test_loss:.4f} | Test accuracy {test_acc:.4f} | Sure {train_time:.1f} sn")

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    plot_history(hist, out / f"{args.name}_egriler.png", args.name)
    torch.save({"state_dict": model.state_dict(), "hidden": args.hidden,
                "dropout": args.dropout}, out / f"{args.name}.pt")
    summary = {
        "hyperparameters": vars(args),
        "final_train_loss": hist["train_loss"][-1],
        "final_val_loss": hist["val_loss"][-1],
        "final_train_acc": hist["train_acc"][-1],
        "final_val_acc": hist["val_acc"][-1],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_time_sec": train_time,
        "history": hist,
    }
    (out / f"{args.name}_sonuc.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
