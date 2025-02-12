import timm
import torch
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

from torch.utils.tensorboard import SummaryWriter

from tqdm import trange

if __name__ == "__main__":
    batch_size = 32
    lr = 3e-5
    epochs = 300

    device = "cuda:0"

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full_dataset = OxfordIIITPet(
        root="./data", split="trainval", transform=transform, download=True
    )

    # 按 80% 训练，20% 验证拆分
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model = timm.create_model("vit_base_patch16_224", pretrained=False)

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter()

    pbar = trange(epochs)
    for epoch in pbar:
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            writer.add_scalars(
                "Loss", {"train": loss.item()}, epoch * len(train_loader) + i
            )

        model.eval()
        with torch.no_grad():
            total, correct = 0, 0
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                output = model(x)
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            writer.add_scalars(
                "Accuracy", {"val": correct / total}, epoch * len(train_loader) + i
            )

        pbar.set_description(f"Epoch {epoch}, val acc: {correct / total}")
