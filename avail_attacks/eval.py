import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        n_total = 0
        n_correct = 0
        for batch in dataloader:
            features = batch[0].to(device)
            labels = batch[1].to(device)
            yhat = model(features)

            n_total += labels.shape[0]
            n_correct += int((labels == torch.argmax(yhat, dim=-1)).sum())

    return float(100.0 * n_correct / n_total)
