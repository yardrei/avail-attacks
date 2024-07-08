from collections import namedtuple
from copy import copy
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pickle
import PIL.Image as Image
import tqdm

from avail_attacks.pgd.sampler import KSamplesHolder, SamplesSelectionMethodology


GradExploderParams = namedtuple('GradExploderParams', ['lr', 'iterations', 'num_labels'])
grad_exploder_defaults = GradExploderParams(lr=0.04, iterations=275, num_labels=10)

class GradExploder:

    def __init__(self,use_grad_exploder : bool, num_idlg_batches: int, batch_size : int, sample_selection : SamplesSelectionMethodology, grad_exploder_params: GradExploderParams = grad_exploder_defaults) -> None:
        if sample_selection is None:
            raise Exception("Sample selection is required")
        num_affected_samples = num_idlg_batches * batch_size
        self._num_exploder_batches = num_idlg_batches
        self._samples_holder = KSamplesHolder(
            k=num_affected_samples,
            keep_largest=sample_selection.is_directional_and_largest(),
        )
        self._use_grad_exploder = use_grad_exploder
        self._iterations = grad_exploder_params.iterations
        self._lr = grad_exploder_params.lr
        self._num_labels = grad_exploder_params.num_labels
        pass

    def get_samples_holder(self):
        return self._samples_holder

    def get_num_exploder_batches(self):
        if not self._use_grad_exploder:
            return 0
        return self._num_exploder_batches
    
    def is_grad_exploder_activated(self):
        return self._use_grad_exploder

    def get_exploder_batch(self, net, data, label, loss_fn):
        data,label = self.run_grad_explosion_v2(net, data, label, loss_fn)
        return (data,label)
    

    def run_grad_explosion_v2(self,net, image, label, loss_fn):
        # net.eval()

        # lr = 0.005
        # reg_lambda = 1000
        # num_dummy = 1
        # num_exp = 1000
        # amplify_magnitude = 10
        # eps=8/255
        # num_labels = 10
        
        best_norm = 0       
        best_data = None
        best_label = None

        use_cuda = torch.cuda.is_available()
        device = 'cuda' if use_cuda else 'cpu'

        tt = transforms.Compose([transforms.ToTensor()])
        tp = transforms.Compose([transforms.ToPILImage()])
        
        dummy_data = torch.clone(image)
        # eps_to_add = rand_eps(dummy_data, eps)
        # dummy_data = dummy_data + eps_to_add
        dummy_data.requires_grad_(True)
        # criterion = nn.CrossEntropyLoss().to(device)
        criterion = loss_fn
        criterion_reduction_none = copy(criterion)  
        criterion_reduction_none.reduction = "none"
        # optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
        optimizer = torch.optim.Adam([dummy_data, ], lr=self._lr)
        # optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)


        loss_per_label = torch.zeros((label.shape[0],self._num_labels))
        for i in range(self._num_labels):
            l = criterion_reduction_none(net(dummy_data),torch.tensor([i]).repeat(label.shape[0]).to(device))
            loss_per_label[:,i] = l

        # print(f"correct label: {label}")
        # print(f"highest loss label: {loss_per_label.argmax()}")


        # measure original grad norm w.r.t original label
        out = net(image)
        y = criterion(out, label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        dy_dx = tuple([item * 1 for item in dy_dx])
        orig_norm = sum([x.norm() for x in dy_dx])
        print('################################# original norm =', orig_norm)
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            
        # label = torch.tensor([loss_per_label.argmax()]).to(device)

        # randomize labels 
        # label = torch.randint(0,num_labels, (label.shape)).to(device)

        label = loss_per_label.argmax(1).to(device)
            
        for iters in range(self._iterations):
            global cur_iters
            cur_iters = iters
            optimizer.zero_grad()
            pred = net(dummy_data)
            dummy_loss = criterion(pred, label)

            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
            total_norm = sum([x.norm() for x in dummy_dy_dx])
            if total_norm > best_norm:
                best_norm = total_norm
                best_data = dummy_data.detach().clone()
                
            grad_diff : float = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                # grad_abs = gx.norm()
                # reg = reg_lambda*((image - dummy_data) ** 2).sum()
                # grad_diff -= (grad_abs - reg) 
                
                grad_abs = gx.norm()
                # reg = reg_lambda*((gt_data - dummy_data) ** 2).sum()
                grad_diff -= grad_abs 
                
            grad_diff.backward()

            optimizer.step()


        # save altered image
        # if altered_norm > best_norm:
        # print('################################# altered norm =', altered_norm)
        # 
        # tp(best_data[0]).save("edited_image.png")
        # tp(image[0]).save("original_image.png")

        pred = net(best_data)
        dummy_loss = criterion(pred, label)
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        total_norm = sum([x.norm() for x in dummy_dy_dx])
        print('Final ret norm   ', total_norm)

        net.train()
        return best_data, label
            
















