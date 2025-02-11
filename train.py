import torch
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

from vit import ViT

batch_size = 32
lr = 3e-5
epochs = 300

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
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
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT(
    output_dim=37,
    d_model=256,
    n_head=8,
    dim_feedforward=512,
    dropout=0.1,
    num_layers=6,
).to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        print(f"Epoch {epoch}, val acc: {correct / total}")
