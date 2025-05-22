import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.utils import SEED


class RandModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.net(x)


def get_rand_dataloader(batch_size=64, shuffle=True, seed=SEED):
    torch.manual_seed(seed)

    inputs = torch.randn(64, 10)
    targets = torch.randint(0, 10, size=(64,), dtype=torch.long)

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
