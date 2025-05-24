import pathlib, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_loaders(root, batch=64, aug=True):
    root = pathlib.Path(root)

    tf_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ]) if aug else tf_val

    base = datasets.ImageFolder(root=root)
    files  = [p for p, _ in base.samples]
    labels = [l for _, l in base.samples]

    tr, tmp, ytr, ytmp = train_test_split(files, labels, test_size=0.30,
                                          stratify=labels, random_state=42)
    val, tst, _, _     = train_test_split(tmp,  ytmp,  test_size=0.50,
                                          stratify=ytmp, random_state=42)

    path2idx = {p: i for i, (p, _) in enumerate(base.samples)}
    def subset(file_list, tf):
        idxs = [path2idx[p] for p in file_list]
        return Subset(datasets.ImageFolder(root=root, transform=tf), idxs)

    dl_tr  = DataLoader(subset(tr,  tf_train), batch_size=batch,
                        shuffle=True,  num_workers=2, pin_memory=True)
    dl_val = DataLoader(subset(val, tf_val),   batch_size=batch,
                        shuffle=False, num_workers=2, pin_memory=True)
    dl_tst = DataLoader(subset(tst, tf_val),   batch_size=batch,
                        shuffle=False, num_workers=2, pin_memory=True)
    return dl_tr, dl_val, dl_tst, base.classes
