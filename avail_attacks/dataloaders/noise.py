from typing import Tuple
import torch
import random
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor


def get_noisy_batch_based_on(dataset: Dataset, batch_size: int) -> Tuple[torch.Tensor]:
    n = len(dataset)
    random_indices = list(range(n))
    random.shuffle(random_indices)
    random_indices = random_indices[:batch_size]
    xs = []
    ys = []
    for idx in random_indices:
        orig_sample = dataset[idx]
        ys.append(orig_sample[1])
        xs.append(torch.rand(size=orig_sample[0].shape))

    return torch.stack(xs, dim=0), torch.Tensor(ys).to(torch.int64)


class DataloaderWithNoise(DataLoader):
    """
    Adds the given number of random (noise) samples to the given base dataloader,
    at the end of the data
    """

    def __init__(self, base_dl: DataLoader, n_noisy_samples: int):
        assert n_noisy_samples >= 0
        self.base_dl = base_dl
        self.n_noisy_samples = n_noisy_samples

    def _get_random_noisy_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ds = self.base_dl.dataset
        i = random.randint(0, len(ds) - 1)
        orig_sample = ds[i]
        label = orig_sample[1]
        X = torch.rand(size=orig_sample[0].shape)
        return X, label

    def __iter__(self):
        bs = self.base_dl.batch_size
        samples_left = self.n_noisy_samples

        for batch in self.base_dl:
            yield batch  # Last batch might be smaller than 'bs', but it doesn't matter too much

        while samples_left > 0:
            next_size = min(bs, samples_left)
            next_batch = get_noisy_batch_based_on(
                self.base_dl.dataset, batch_size=next_size
            )
            yield next_batch
            samples_left -= next_size
