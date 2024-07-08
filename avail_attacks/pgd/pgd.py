from __future__ import annotations
import torch
import copy
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from ..dataloaders.noise import get_noisy_batch_based_on
from ..nes import estimate_model_grad_for
from ..train import train_iter
from .sampler import SamplesSelectionMethodology, KSamplesHolder


class DataloaderPGD(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        model: nn.Module,
        loss_func: nn.CrossEntropyLoss,
        sample_selection_methodology: SamplesSelectionMethodology,
        is_targeted: int,
        number_of_classes: int,
        transfer_learning: int,
        transfer_optimizer: torch.optim.Optimizer,
        epochs: int = 5,
        step_size: float = 0.1,
        fraction_affected_samples: float = 0,
        use_cw: bool = False,
        cw_c_constant: float = 100.0,
        is_black_box: bool = False,
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
        self.use_cw = use_cw
        self.cw_c_constant = cw_c_constant
        self.model = model
        self.loss_func = loss_func
        self.epochs = epochs
        self.step_size = step_size
        self.dataset_length = len(self.dataset)
        self.number_of_classes = number_of_classes
        self.num_affected_samples = int(
            np.ceil(self.dataset_length * fraction_affected_samples)
        )
        self.sample_selection_methodology = sample_selection_methodology
        self.is_targeted = is_targeted
        self.transfer_learning = transfer_learning
        self.transfer_optimizer = transfer_optimizer
        self.model_device = next(self.model.parameters()).device
        print(
            f"Using {'CW' if self.use_cw else 'PGD'} {'Targeted ' + str(is_targeted) if is_targeted else 'Untargeted'} attack"
        )
        print(
            f"Will be adding/mutating {self.num_affected_samples} samples using {self.sample_selection_methodology}"
        )
        print()

        self.count_dict = dict()
        self.count_dict_predict = dict()

        self.last_epoch_num_adv_samples = 0
        self.last_epoch_adv_samples_l2_sum = 0.0
        self.last_epoch_adv_samples_l2_mean = 0.0
        self.is_black_box = is_black_box

    def __iter__(self):
        self.last_epoch_num_adv_samples = 0
        self.last_epoch_adv_samples_l2_sum = 0.0
        self.last_epoch_adv_samples_l2_mean = 0.0

        samples_holder = KSamplesHolder(
            k=self.num_affected_samples,
            keep_largest=self.sample_selection_methodology.is_directional_and_largest(),
        )
        for batch in super().__iter__():
            yield batch
            # Keep track of the worst / best samples
            x = batch[0].to(self.model_device)
            y = batch[1].to(self.model_device)

            if self.sample_selection_methodology.is_loss_based():
                self.model.eval()
                yhat = self.model(x)
                self.model.train()
                loss_f = copy.copy(self.loss_func)  # We need loss without reduction
                loss_f.reduction = "none"
                loss: torch.Tensor = loss_f(yhat, y)
                samples_holder.push_batch(batch, metric_values=loss)

            elif self.sample_selection_methodology.is_grad_based():
                self.model.eval()
                params = next(iter(self.model.parameters()))
                for i in range(len(x)):
                    yhat = self.model(x[i : i + 1])
                    sample_loss = self.loss_func(yhat, y[i : i + 1])
                    grad = torch.autograd.grad(sample_loss, params)
                    total_grad = 0.0
                    for lay in grad:
                        lay: torch.Tensor
                        total_grad += float(lay.flatten().abs().sum().detach().cpu())
                    samples_holder.push_sample(
                        metric_value=total_grad, sample=(x[i], y[i])
                    )
                self.model.train()

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

        if (
            self.sample_selection_methodology
            == SamplesSelectionMethodology.take_from_next_epoch
        ):
            pgd_batches_generator = super().__iter__()
        elif (
            self.sample_selection_methodology
            == SamplesSelectionMethodology.create_new_from_noise
        ):

            def rand_samples_yielder():
                while True:
                    yield get_noisy_batch_based_on(
                        dataset=self.dataset, batch_size=self.batch_size
                    )

            pgd_batches_generator = rand_samples_yielder()
        elif (
            self.sample_selection_methodology.is_grad_based()
            or self.sample_selection_methodology.is_loss_based()
        ):

            def samples_yielder_from_k_samples_holder():
                xs = [s[1][0] for s in samples_holder.samples]
                ys = [s[1][1] for s in samples_holder.samples]
                bs = self.batch_size
                n = len(ys)
                i = 0
                while i <= n:
                    batch_xs = torch.stack(xs[i : i + bs], dim=0)
                    batch_ys = torch.stack(ys[i : i + bs], dim=0)
                    yield batch_xs, batch_ys
                    i += bs

            pgd_batches_generator = samples_yielder_from_k_samples_holder()
        else:
            raise NotImplementedError(f"PGD with {self.sample_selection_methodology}")

        samples_left = self.num_affected_samples
        while samples_left > 0:
            size = min(self.batch_size, samples_left)
            batch = next(pgd_batches_generator)
            update_counts(batch[1][:size], self.count_dict)
            if self.is_targeted > 0:
                if self.is_targeted == 1:
                    batch_pgd = self.get_attack_data(
                        batch[0][:size], batch[1][:size], True
                    )
                    add_to_y = torch.randint(1, self.number_of_classes, (size,))
                    new_y = torch.remainder(
                        add_to_y + batch_pgd[1].to(add_to_y.device), 10
                    )
                    new_batch = (batch_pgd[0].detach(), new_y.detach())
                elif self.is_targeted == 2:
                    new_batch = self._targeted_attack_optimize_over_original_label(
                        batch, size
                    )
                    new_batch = new_batch[0].detach(), new_batch[1].detach()
                elif self.is_targeted == 3:
                    new_batch = self._targeted_attack_optimize_over_wrong_label(
                        batch, size, return_orig_label=False
                    )
                    new_batch = new_batch[0].detach(), new_batch[1].detach()
                elif self.is_targeted == 4:
                    new_batch = self._targeted_attack_optimize_over_wrong_label(
                        batch, size, return_orig_label=True
                    )
                    new_batch = new_batch[0].detach(), new_batch[1].detach()
                else:
                    exit(1)
                yield new_batch

            else:
                new_batch = self.get_attack_data(batch[0][:size], batch[1][:size])

                self.model.eval()
                y_prediction = torch.argmax(
                    self.model(new_batch[0].to(self.model_device)), dim=1
                )
                self.model.train()
                update_counts(y_prediction, self.count_dict_predict)

                yield new_batch[0].detach(), new_batch[1].detach()
            samples_left -= len(new_batch[0])

            if self.transfer_learning == 2:
                train_iter(
                    model=self.model,
                    inputs=new_batch[0].detach().to(self.model_device),
                    labels=new_batch[1].detach().to(self.model_device),
                    optimizer=self.transfer_optimizer,
                    loss_fn=self.loss_func,
                    grad_exploder=None,
                    use_grad_clipping=False,
                    grad_clipping_value=0,
                )

    def get_attack_data(self, x, y, is_targeted=False):
        # Don't record model gradients for the attack iterations
        self.model.eval()
        phantom_optim = torch.optim.Adadelta(
            self.model.parameters()
        )  # Used to zero model grads later

        if self.use_cw:
            attack_x, attack_y = self.get_cw_data(x, y, is_targeted)
        else:
            attack_x, attack_y = self.get_pgd_data(x, y, is_targeted)
        phantom_optim.zero_grad()
        self.model.train()

        diff = attack_x.to(self.model_device) - x.to(self.model_device)

        adv_l2_norm = torch.norm(diff, p=2, dim=(-1, -2))
        self.last_epoch_num_adv_samples += int(attack_x.shape[0])
        self.last_epoch_adv_samples_l2_sum += float(adv_l2_norm.sum())
        self.last_epoch_adv_samples_l2_mean = (
            self.last_epoch_adv_samples_l2_sum / self.last_epoch_num_adv_samples
        )

        return attack_x, attack_y

    def get_model_grad_for(self, x, y, loss) -> torch.Tensor:
        if not self.is_black_box:
            return torch.autograd.grad(torch.sum(loss), x)
        else:
            return estimate_model_grad_for(
                model=self.model, x=x, y=y, loss=loss, device=self.model_device
            )

    def get_pgd_data(self, x, y, is_targeted=False):
        device = self.model_device
        y = y.to(device)
        projected_x: torch.tensor = x.clone().detach().to(device)

        for i in range(self.epochs):
            projected_x.requires_grad_(True)
            y_prediction = self.model(projected_x).to(device)

            # Finding loss
            loss = self.loss_func(y_prediction, y)

            # Gradient descent step
            grad = self.get_model_grad_for(x=projected_x, y=y, loss=loss)
            sign_grad = torch.sign(grad[0])

            if is_targeted:
                step_x = projected_x - (self.step_size * sign_grad)
            else:
                step_x = projected_x + (self.step_size * sign_grad)

            projected_x = torch.clamp(step_x, 0, 1)

        return projected_x.clone().detach().to(device), y.to(device)

    def get_cw_data(self, x, y, is_targeted=True):
        """
        Performs the Carlini-Wagner attack
        """
        assert is_targeted, "CW not implemented for untargeted"
        device = self.model_device
        y = y.to(device)
        x: torch.Tensor = x.clone().detach().to(device)

        def loss_f(yhat, y):
            # We use "f_6" from the paper, which performed best
            ypred = torch.gather(yhat, dim=1, index=y.unsqueeze(-1))
            res: torch.Tensor = torch.argmax(yhat, dim=1).unsqueeze(-1) - ypred
            return res.clamp(min=0)  # max(0, res)

        # In order to satisfy box constraints, we optimize over omega
        omega = torch.zeros(size=x.shape).to(device)
        optim = torch.optim.Adagrad(params=[omega], lr=0.3)
        omega.requires_grad_(True)

        def get_delta(omega) -> torch.Tensor:
            return ((torch.tanh(omega) + 1) / 2) - x

        for i in range(self.epochs):
            optim.zero_grad()

            assert not self.is_black_box, "CW not implemented for Black box"
            delta = get_delta(omega)
            yhat = self.model(x + delta)

            # Finding loss
            # We use l2 as suggested by the paper
            delta_l2 = torch.norm(delta, p=2, dim=(-1, -2))
            assert is_targeted
            loss = delta_l2 + self.cw_c_constant * loss_f(yhat, y)
            loss = loss.sum()

            # Gradient descent step
            loss.backward()
            optim.step()
        self.model.train()

        delta = get_delta(omega)
        assert bool((x + delta >= 0.0).all())
        assert bool((x + delta <= 1.0).all())
        return x + delta, y

    def _targeted_attack_optimize_over_wrong_label(
        self, batch, size, return_orig_label: bool = False
    ):
        orig_label = batch[1][:size]
        add_to_y = torch.randint(1, self.number_of_classes, (size,))
        new_y = torch.remainder(orig_label + add_to_y, self.number_of_classes)
        new_batch = self.get_attack_data(batch[0][:size], new_y, True)
        if return_orig_label:
            return new_batch[0], orig_label
        else:
            return new_batch[0], new_y

    def _targeted_attack_optimize_over_original_label(self, batch, size):
        new_batch = self.get_attack_data(batch[0][:size], batch[1][:size], True)
        self.model.eval()
        y_prediction = self.model(new_batch[0].to(self.model_device))
        self.model.train()
        new_y = torch.argmin(y_prediction, dim=1)
        return new_batch[0], new_y


def update_counts(tensor, count_dict):
    # Flatten the tensor
    flat_tensor = tensor.view(-1)

    # Count the occurrences of scalar values
    unique_values, counts = torch.unique(flat_tensor, return_counts=True)

    # Update the count_dict with the new counts
    for value, count in zip(unique_values, counts):
        value = value.item()
        count = count.item()
        if value in count_dict:
            count_dict[value] += count
        else:
            count_dict[value] = count
    return count_dict
