import torch

def BCE(predictions, labels):
    criterion = torch.nn.BCELoss()

    loss = criterion(predictions, labels)
    return loss

def MSE(predictions, labels):
    criterion = torch.nn.MSELoss()

    loss = criterion(predictions, labels)
    return loss