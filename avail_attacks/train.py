from typing import List, Tuple
from dataclasses import dataclass
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd


from .eval import accuracy
from avail_attacks.grad_explosion.iDLG import *


@dataclass
class SingleEpochEvals:
    test_acc: float
    num_adv_samples: int
    mean_l2_of_adv_samples: int
    avg_grad_norm: float 
    adv_avg_grad_norm: float 


@dataclass
class TrainRes:
    per_epoch: List[SingleEpochEvals]

    def to_df(self) -> pd.DataFrame:
        rows = [epoch_res.__dict__ for epoch_res in self.per_epoch]
        df = pd.DataFrame(rows)
        df.index.rename("epoch", inplace=True)
        return df


def train_iter(
    model,
    inputs,
    labels,
    optimizer,
    loss_fn,
    use_grad_clipping,
    grad_clipping_value,
    grad_exploder = None
):
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    output = model(inputs)
    loss = loss_fn(output, labels)

    # Backward
    loss.backward()

    # Clip
    if use_grad_clipping:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping_value)


    # Step
    optimizer.step()


def train(
    model: nn.Module,
    train_dl: DataLoader,
    test_dl,
    epochs,
    loss_fn,
    optimizer,
    lr_scheduler,
    device: torch.device,
    use_grad_clipping: bool,
    grad_clipping_value: float,
) -> Tuple[nn.Module, TrainRes]:
    res = TrainRes([])

    # Add initial evaluations
    model.eval()
    res.per_epoch.append(
        SingleEpochEvals(
            test_acc=accuracy(model=model, dataloader=test_dl, device=device),
            num_adv_samples=0,
            mean_l2_of_adv_samples=0.0,
            avg_grad_norm=0,
            adv_avg_grad_norm=0
        )
    )

    # train
    sbar = tqdm.tqdm(list(range(epochs)), position=0, desc="Epoch")
    for epoch in sbar:  # loop over the dataset multiple times
        avg_grad_norms = []
        adv_avg_grad_norms = []
        model.train()
        for i, data in enumerate(train_dl, 0):
            # get inputs and labels
            inputs, labels = data[0].to(device), data[1].to(device)

            train_iter(
                model,
                inputs,
                labels,
                optimizer,
                loss_fn,
                use_grad_clipping,
                grad_clipping_value,
            )            

            grad_list = [param.grad.norm() for param in model.parameters()]
            avg_grad_norm = sum(grad_list) / len(grad_list)
            avg_grad_norms.append(avg_grad_norm)

        lr_scheduler.step()

        # Epoch finished, add evaluations
        model.eval()

        num_adv_samples = getattr(train_dl, "last_epoch_num_adv_samples", 0)
        mean_l2_of_adv_samples = getattr(
            train_dl, "last_epoch_adv_samples_l2_mean", 0.0
        )
        res.per_epoch.append(
            SingleEpochEvals(
                test_acc=accuracy(model=model, dataloader=test_dl, device=device),
                num_adv_samples=num_adv_samples,
                mean_l2_of_adv_samples=mean_l2_of_adv_samples,
                avg_grad_norm= (sum(avg_grad_norms) / len(avg_grad_norms)) if len(avg_grad_norms) > 0 else 0,
                adv_avg_grad_norm= (sum(adv_avg_grad_norms) / len(adv_avg_grad_norms)) if len(adv_avg_grad_norms) > 0 else 0
            )
        )

    # done
    return model, res
