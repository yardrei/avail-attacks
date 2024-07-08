from enum import Enum
import bisect
import torch
import copy


class SamplesSelectionMethodology(Enum):
    take_from_next_epoch = 1
    create_new_from_noise = 2
    largest_abs_grad = 3
    smallest_abs_grad = 4
    largest_model_loss = 5
    smallest_model_loss = 6

    def is_directional_and_largest(self) -> bool:
        return self in [
            SamplesSelectionMethodology.largest_abs_grad,
            SamplesSelectionMethodology.largest_model_loss,
        ]

    def is_loss_based(self) -> bool:
        return self in [
            SamplesSelectionMethodology.smallest_model_loss,
            SamplesSelectionMethodology.largest_model_loss,
        ]

    def is_grad_based(self) -> bool:
        return self in [
            SamplesSelectionMethodology.largest_abs_grad,
            SamplesSelectionMethodology.smallest_abs_grad,
        ]


class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)


class KSamplesHolder:
    """
    Holds Top-K samples, sorted by the given metric
    """

    def __init__(self, k: int, keep_largest: bool = True) -> None:
        self.samples = []
        self.keep_largest = keep_largest
        self.k = k
        assert k > 0

    def __getitem__(self, i):
        return self.samples[i][0]  # Only index by metric

    def __len__(self):
        return len(self.samples)

    def get_batch(self, start: int, end: int):
        sub_samples = self.samples[start:end]
        inputs = torch.stack([s[1][0] for s in sub_samples])
        labels = torch.stack([s[1][1] for s in sub_samples])

        return inputs, labels

    def get_loss_with_reduction_none(self, loss_fn):
        loss_f = copy.copy(loss_fn)  # We need loss without reduction
        loss_f.reduction = "none"
        return loss_f

    def push_sample(self, metric_value: float, sample) -> None:
        idx = bisect.bisect(self, metric_value)
        self.samples.insert(idx, (metric_value, sample))

        if self.keep_largest:
            self.samples = self.samples[-self.k :]
        else:
            self.samples = self.samples[: self.k]

    def push_batch(self, batch, metric_values) -> None:
        for i in range(len(metric_values)):
            self.push_sample(
                float(metric_values[i].detach().cpu()),
                (batch[0][i], batch[1][i]),
            )

