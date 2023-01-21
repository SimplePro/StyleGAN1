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