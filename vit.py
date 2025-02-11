import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=256, kernel_size=16, stride=16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)  # (B, C, H, W) -> (B, D, CH, CW)
        x = x.flatten(2)  # (B, D, CH, CW) -> (B, D, N)
        x = x.transpose(1, 2)  # (B, D, N) -> (B, N, D)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        output_dim: int,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.patch_embedding = PatchEmbedding()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.position_embedding = nn.Parameter(torch.randn(1, 197, d_model))

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.Tanh(),
            nn.Linear(d_model * 4, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x += self.position_embedding

        x = self.transformer(x)
        output = self.classifier(x[:, 0])

        return output
