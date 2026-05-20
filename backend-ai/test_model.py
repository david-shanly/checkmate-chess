import torch
from model import ChessNet

# create model
model = ChessNet()

# fake chess input
x = torch.randn(1, 13, 8, 8)

# run forward pass
y = model(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)