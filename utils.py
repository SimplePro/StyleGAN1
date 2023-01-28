from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_loader(image_size, dataset_root, batch_size):
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)),
         transforms.ToTensor(),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
         )
        ]
    )
    dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return loader, dataset


if __name__ == '__main__':
    loader, _ = get_loader(image_size=128, dataset_root="/home/kdhsimplepro/kdhsimplepro/AI/ffhq/", batch_size=16)
    print(len(loader), 52000 / len(loader))