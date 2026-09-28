import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )

        self.model = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )

    def forward(self, x):
        return self.model(x)


def create_workload():
    model = TinyTransformer()

    x = torch.randn(
        2,      # batch
        128,    # sequence length
        512,    # hidden dimension
    )

    return model, x
