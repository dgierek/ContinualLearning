import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from data.dataset import load_synthetic_k0_E0_data
from models.small_net import CVMidNet
from training.trainer import evaluate


class EWC:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.device = device

        self.params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()

        for x, y in dataloader:

            x = x.to(self.device)
            y = y.to(self.device)

            self.model.zero_grad()

            output = self.model(x)

            loss = nn.CrossEntropyLoss()(output, y)

            loss.backward()

            for n, p in self.model.named_parameters():

                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

        for n in fisher:
            fisher[n] /= len(dataloader)

        return fisher

    def penalty(self, model):

        loss = 0

        for n, p in model.named_parameters():

            if n in self.fisher:

                loss += (
                    self.fisher[n]
                    * (p - self.params[n]) ** 2
                ).sum()

        return loss


epochs = 40
lambda_ewc = 1

(substance0_train, substance1_train, substance2_train, substance3_train,
 substance0_test, substance1_test, substance2_test, substance3_test) = load_synthetic_k0_E0_data()

loader_test_A = DataLoader(ConcatDataset([substance0_test, substance1_test]), batch_size=32, shuffle=True)
loader_A = DataLoader(ConcatDataset([substance0_train, substance1_train]), batch_size=32, shuffle=True)
loader_B = DataLoader(substance2_train, batch_size=32, shuffle=True)


model = CVMidNet()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):

    for x, y in loader_A:

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(pred, y)

        loss.backward()

        optimizer.step()

print('Accuracy on task A:', evaluate(model, ConcatDataset([substance0_test, substance1_test]))[1])
print('Accuracy on task B:', evaluate(model, substance2_test)[1])

device = "cuda" if torch.cuda.is_available() else "cpu"

ewc = EWC(
    model=model,
    dataloader=loader_test_A,
    device=device
)

for x, y in loader_B:

    optimizer.zero_grad()

    pred = model(x)

    classification_loss = criterion(
        pred,
        y
    )

    ewc_loss = ewc.penalty(model)

    loss = classification_loss + \
           lambda_ewc * ewc_loss

    loss.backward()

    optimizer.step()

print('Accuracy on task A:', evaluate(model, ConcatDataset([substance0_test, substance1_test]))[1])
print('Accuracy on task B:', evaluate(model, substance2_test)[1])