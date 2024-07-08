import torch
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np


class DataloaderWhite(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        number_of_classes: int,
        fraction_affected_samples: float = 0,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        assert fraction_affected_samples >= 0
        self.fraction_affected_samples = fraction_affected_samples
        self.number_of_classes = number_of_classes
        self.batch_shape = next(super().__iter__())[0].shape
        class_frequency, self.dataset_length = get_frequency(dataset, self.number_of_classes)
        self.num_affected_samples = int(
            math.ceil(self.dataset_length * fraction_affected_samples)
        )
        self.frequency = np.array(class_frequency)/self.dataset_length
        print(
            f"Adds {self.num_affected_samples} white images"
        )

    def __iter__(self):
        for batch in super().__iter__():
            yield batch

        number_of_samples = self.num_affected_samples
        samples_done = 0

        labels_arr = np.arange(self.number_of_classes)
        class_frequency = np.ceil(self.frequency * number_of_samples).astype(int)
        repeated_labels = np.repeat(labels_arr, class_frequency)
        np.random.shuffle(repeated_labels)
        labels = torch.tensor(repeated_labels[:number_of_samples], dtype=torch.int64)

        while number_of_samples > samples_done:
            size = min(self.batch_size, number_of_samples-samples_done)
            images = torch.ones(self.batch_shape)[:size]
            yield images, labels[samples_done: samples_done + size]
            samples_done += size


def get_frequency(dataset: Dataset, number_of_classes: int):
    labels = [0] * number_of_classes
    i = 0
    for _, target in dataset:
        i += 1
        labels[target] += 1
    return labels, i
