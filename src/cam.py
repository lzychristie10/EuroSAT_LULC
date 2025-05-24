import argparse, random, torch, matplotlib.pyplot as plt
from torchvision import models, transforms
from pathlib import Path
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset import get_loaders, IMAGENET_MEAN, IMAGENET_STD

Path("results").mkdir(exist_ok=True)

inv_norm = transforms.Normalize(
    mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std =[ 1/s for s      in  IMAGENET_STD])

def to_rgb(t):          # Tensor → 0-1 RGB (HWC)
    return inv_norm(t).clamp(0,1).permute(1,2,0).cpu().numpy()

def main(root, weights, num):
    _, _, dl_test, classes = get_loaders(root, aug=False)

    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    cam = GradCAMPlusPlus(model, target_layers=[model.layer3[-1]])

    samples = random.sample(range(len(dl_test.dataset)), num)
    fig, ax = plt.subplots(num, 2, figsize=(6, 3*num))

    for row, idx in enumerate(samples):
        img, true_lbl = dl_test.dataset[idx]
        rgb = to_rgb(img)
        pred = model(img.unsqueeze(0)).argmax(1).item()

        cam_map = cam(img.unsqueeze(0),
                      targets=[ClassifierOutputTarget(pred)])[0]
        ax[row,0].imshow(rgb); ax[row,0].set_title(
            f"T:{classes[true_lbl]}  P:{classes[pred]}", fontsize=9)
        ax[row,0].axis("off")

        ax[row,1].imshow(show_cam_on_image(rgb, cam_map, use_rgb=True))
        ax[row,1].axis("off")

    plt.tight_layout()
    plt.savefig("results/gradcam_examples.png", dpi=200)
    print("Saved results/gradcam_examples.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    default="data/EuroSAT_RGB")
    ap.add_argument("--weights", default="results/best_eurosat_resnet50.pt")
    ap.add_argument("--num", type=int, default=5)
    args = ap.parse_args()
    main(args.root, args.weights, args.num)
