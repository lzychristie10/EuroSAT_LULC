import argparse, copy, torch, torch.nn as nn
from torchvision import models
from torch.optim import AdamW, lr_scheduler
from codecarbon import EmissionsTracker
from dataset import get_loaders

def main(root):
    dl_train, dl_val, _, _ = get_loaders(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim   = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched   = lr_scheduler.CosineAnnealingLR(optim, T_max=15)

    best_acc = 0; best_wts = None; no_imp = 0
    tracker = EmissionsTracker(project_name="EuroSAT", output_dir=".", log_level="error")
    tracker.start()

    for epoch in range(25):
        # ---- train ----
        model.train(); correct = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            preds = model(xb); loss = loss_fn(preds, yb)
            loss.backward(); optim.step()
            correct += (preds.argmax(1) == yb).sum().item()
        tr_acc = correct / len(dl_train.dataset)

        # ---- val ----
        model.eval(); correct = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                correct += (preds.argmax(1) == yb).sum().item()
        val_acc = correct / len(dl_val.dataset)
        print(f"[{epoch+1:02d}] train {tr_acc:.3f} | val {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc, best_wts, no_imp = val_acc, copy.deepcopy(model.state_dict()), 0
            torch.save(best_wts, "results/best_eurosat_resnet50.pt")
        else:
            no_imp += 1
            if no_imp == 5:
                print("Early stopping"); break
        sched.step()

    tracker.stop()
    print("Best val acc:", best_acc)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/EuroSAT_RGB")
    args = ap.parse_args()
    main(args.root)
