import torch


def project_both(res,x,eps):
    tmp = project(res)
    return project_eps_ball(tmp, x, eps)

def project(res):
    # max_part = torch.maximum(res, torch.zeros_like(res))
    # min_part = torch.minimum(max_part, torch.ones_like(max_part))
    return torch.clamp(res, 0, 1)

def project_eps_ball(res, x, eps):
    return torch.clamp(res, x - eps, x + eps)

class GradEvaluator:
    # used to compute gradient, either white box or black box
    def __init__(self, is_black_box, loss_func, nes_k = 200, eps=8/255, sigma=1/255):
        self._is_black_box = is_black_box
        self._nes_k = nes_k
        self._eps=eps
        self._sigma = sigma
        self._loss_func = loss_func

    def evaluate_grad(self, model, x,y, outputs, inputs, create_graph = False):
        if not self._is_black_box: # white box 
            return torch.autograd.grad(outputs, inputs, create_graph=create_graph)
        
        delta = torch.randn(self._nes_k, *x.shape)

        #anti
        # other_delta_half = - 1 * torch.flip(delta,[0])
        delta = torch.cat([delta, -1 * delta])

        total_sum_batched = 0
        # print(f"delta shape {delta.shape}")

        for m in range(delta.shape[0]):
            cur_delta = delta[m].to(x.device)
            reshaped_delta = cur_delta
            # reshaped_delta = cur_delta
            theta = x + self._sigma * reshaped_delta.to(x.device)

            # clamp?
            theta = project_both(theta, x, self._eps)

            area_preds = model(theta).detach()
            theta_loss = self._loss_func(area_preds, y).detach()
            theta_loss_repeated = torch.stack([theta_loss[s].repeat(3,32,32) for s in range(theta_loss.shape[0])])
            inner_sum = reshaped_delta * theta_loss_repeated
            total_sum_batched += inner_sum

        grad_approx = ( 1 /(self._sigma * self._nes_k * 2) ) * total_sum_batched

        return grad_approx