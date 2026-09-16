import torch
from models.small_net import CVMidNet

model = CVMidNet(in_channels=2, num_classes=4)
torch.save(model, "cvmidnet.pt")
