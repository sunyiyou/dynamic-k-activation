import os
import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets

def places365_loader(settings, dir, batch_size=None, shuffle=False, data_augment=False):
    if batch_size is None:
        batch_size = settings.BATCH_SIZE
    d = os.path.join(settings.DATASET_PATH, dir)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if data_augment:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomSizedCrop(settings.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(settings.IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])
    loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(d, transform),
        batch_size=batch_size, shuffle=shuffle,
        num_workers=settings.WORKERS, pin_memory=True)
    return loader
