from typing import Tuple
import torch
import random
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from avail_attacks.grad_explosion.iDLG import GradExploder, GradExploderParams
from avail_attacks.pgd.sampler import SamplesSelectionMethodology

class DataloaderExploder(DataLoader):
    """
    Adds the given number of random (noise) samples to the given base dataloader,
    at the end of the data
    """
    
    def __init__(self,base_dl: DataLoader, model : torch.nn.Module, shuffle: bool, pin_memory: bool, num_workers: int, 
                 use_grad_exploder : bool, num_idlg_batches: int, batch_size : int, sample_selection : SamplesSelectionMethodology, loss_fn) -> None:
        super().__init__(
            base_dl,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        self.grad_exploder = GradExploder(use_grad_exploder, num_idlg_batches, batch_size, sample_selection)
        self.samples_holder = self.grad_exploder.get_samples_holder()
        self.model = model
        self.base_dl = base_dl   
        self.loss_fn = loss_fn
        self.model_device = next(self.model.parameters()).device

    def __iter__(self):
        batches_for_later = []
        for i,batch in enumerate(super().__iter__()):
            # yield batch  # Last batch might be smaller than 'bs', but it doesn't matter too much

            inputs, labels = batch[0].to(self.model_device), batch[1].to(self.model_device)
            output = self.model(inputs)
            loss = self.loss_fn(output, labels)
            loss_red_none = self.samples_holder.get_loss_with_reduction_none(self.loss_fn)
            loss_per_item = loss_red_none(output, labels)
            self.samples_holder.push_batch((inputs, labels), loss_per_item)

            batches_for_later.append(batch)
        
        for batch in batches_for_later:
            yield batch

        
        for i in range(self.grad_exploder.get_num_exploder_batches()):
            batch_size = self.batch_size
            inputs, labels = self.samples_holder.get_batch(
                i * batch_size, (i + 1) * batch_size
            )
            inputs, labels = self.grad_exploder.get_exploder_batch(
                self.model, inputs, labels, self.loss_fn
            )  # note : computes gradient on weights

            yield (inputs,labels)

