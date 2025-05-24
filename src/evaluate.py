import argparse, torch, seaborn as sns, matplotlib.pyplot as plt, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models
from pathlib import Path
from dataset import get_loaders

Path("results").mkdir(exist_ok=True)

def main(root, weights):
    _, _, dl_test, classes = get_loaders(root, aug=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            preds.append(model(xb.to(device)).argmax(1).cpu())
            labels.append(yb)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = (preds == labels).mean()
    print(f"Test acc: {acc:.3f}")

    pd.DataFrame(classification_report(labels, preds, target_names=classes, output_dict=True)
                ).to_csv("results/classification_report.csv")
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.tight_layout(); plt.savefig("results/confusion_matrix.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    default="data/EuroSAT_RGB")
    ap.add_argument("--weights", default="results/best_eurosat_resnet50.pt")
    args = ap.parse_args()
    main(args.root, args.weights)
