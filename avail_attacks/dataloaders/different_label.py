import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import bisect
from enum import Enum
from ..train import train_iter


class WrongSelectionMethodology(Enum):
    take_from_next_epoch = 1
    highest_confidence = 2
    random = 3


class TopEachClass:
    """
    Holds Top-K samples for each class
    """

    def __init__(self, num_in_each_class: list, batch_size: int = 128) -> None:
        self.samples = {clss: [[], []] for clss in range(len(num_in_each_class))}
        self.num_in_each_class = num_in_each_class
        self.batch_size = batch_size

    def __iter__(self):
        data_tensor, labels_tensor = self.get_tensors()
        perm = torch.randperm(data_tensor.shape[0])
        return TopClassIterator(data_tensor[perm], labels_tensor[perm], self.batch_size)

    def get_tensors(self):
        stacked_tensors = list()
        labels = list()

        for label, list_of_list in self.samples.items():
            stacked_tensors.extend(list_of_list[1])
            labels.extend([label] * len(list_of_list[1]))

        data_tensor = torch.stack(stacked_tensors)
        labels_tensor = torch.tensor(labels)

        return data_tensor, labels_tensor

    def add_score(self, cls: int, sample, score):
        if cls in self.samples:
            index = bisect.bisect(self.samples[cls][0], score)
            if index == self.num_in_each_class[cls]:
                return
            self.samples[cls][0].insert(index, score)
            self.samples[cls][1].insert(index, sample)

            if len(self.samples[cls][0]) > self.num_in_each_class[cls]:
                self.samples[cls][0].pop()
                self.samples[cls][1].pop()

    def push_batch(self, batch, scores) -> None:
        for i in range(scores.shape[0]):
            self.add_score(batch[1][i].item(), batch[0][i], float(scores[i]))


class TopClassIterator:
    def __init__(
        self, data_tensor: torch.Tensor, labels_tensor: torch.Tensor, batch_size: int
    ):
        self.data_tensor = data_tensor
        self.labels_tensor = labels_tensor
        self.len = labels_tensor.shape[0]
        self.batch_size = batch_size
        self.curr_index = 0

    def __len__(self):
        return math.ceil(self.len / self.batch_size)

    def __next__(self):
        if self.curr_index >= self.len:
            raise StopIteration

        size = min(self.len - self.curr_index, self.batch_size)

        data = self.data_tensor[self.curr_index : self.curr_index + size]
        labels = self.labels_tensor[self.curr_index : self.curr_index + size]
        self.curr_index += size
        return data, labels


class DataloaderDifferentLabel(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        sample_selection_methodology: WrongSelectionMethodology,
        model: nn.Module,
        loss_func: nn.CrossEntropyLoss,
        number_of_classes: int,
        transfer_learning: int,
        transfer_optimizer: torch.optim.Optimizer,
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
        self.model = model
        self.loss_func = loss_func
        self.sample_selection_methodology = sample_selection_methodology
        assert hasattr(self.dataset, "__len__")
        self.dataset_length = len(self.dataset)
        self.number_of_classes = number_of_classes
        self.num_affected_samples = int(
            math.ceil(self.dataset_length * fraction_affected_samples)
        )
        self.transfer_learning = transfer_learning
        self.transfer_optimizer = transfer_optimizer
        self.model_device = next(self.model.parameters()).device
        print(f"Different label of {self.num_affected_samples} samples")

    def __iter__(self):
        for batch in super().__iter__():
            yield batch

            if self.transfer_learning:
                train_iter(
                    model=self.model,
                    inputs=batch[0].to(self.model_device),
                    labels=batch[1].to(self.model_device),
                    optimizer=self.transfer_optimizer,
                    loss_fn=self.loss_func,
                    grad_exploder=None,
                    use_grad_clipping=False,
                    grad_clipping_value=0,
                )

        samples_left = self.num_affected_samples

        # Choose the data to change
        if (
            self.sample_selection_methodology
            == WrongSelectionMethodology.take_from_next_epoch
            or self.sample_selection_methodology == WrongSelectionMethodology.random
        ):
            batches_generator = super().__iter__()
        elif (
            self.sample_selection_methodology
            == WrongSelectionMethodology.highest_confidence
        ):
            top_handler = self.init_top_of_each_class()
            for batch in super().__iter__():
                x, y = batch[0].to(self.model_device), batch[1].to(self.model_device)
                self.model.eval()
                predicted_y = self.model(x).detach().cpu()
                self.model.train()
                top_handler.push_batch(
                    batch, predicted_y[torch.arange(predicted_y.shape[0]), y.cpu()]
                )
            batches_generator = top_handler.__iter__()
        else:
            exit(1)

        while samples_left > 0:
            batch = next(batches_generator)
            size = min(self.batch_size, samples_left)

            # Random or worst label
            if self.sample_selection_methodology == WrongSelectionMethodology.random:
                add_to_y = torch.randint(1, self.number_of_classes, (size,))
                new_label = torch.remainder(add_to_y + batch[1][:size].to(add_to_y.device), 10)
            else:
                self.model.eval()
                y_prediction = self.model(batch[0][:size].to(self.model_device)).to(
                    batch[0].device
                )
                self.model.train()
                new_label = torch.argmin(y_prediction, dim=1)

            yield batch[0][:size], new_label.detach()
            samples_left -= size

            # Training the transfer model also on the wrong data
            if self.transfer_learning == 2:
                train_iter(
                    model=self.model,
                    inputs=batch[0][:size].to(self.model_device),
                    labels=new_label.detach().to(self.model_device),
                    optimizer=self.transfer_optimizer,
                    loss_fn=self.loss_func,
                    grad_exploder=None,
                    use_grad_clipping=False,
                    grad_clipping_value=0,
                )

    def init_top_of_each_class(self):
        freq, data_len = get_frequency(self.dataset, self.number_of_classes)
        num_in_each_class = [
            math.floor(x / data_len * self.num_affected_samples) for x in freq
        ]
        i = 0
        while sum(num_in_each_class) < self.num_affected_samples:
            num_in_each_class[i] += 1
            i += 1
        return TopEachClass(
            num_in_each_class=num_in_each_class, batch_size=self.batch_size
        )


def get_frequency(dataset: Dataset, number_of_classes: int):
    labels = [0] * number_of_classes
    i = 0
    for _, target in dataset:
        i += 1
        labels[target] += 1
    return labels, i
