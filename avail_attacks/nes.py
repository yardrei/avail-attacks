import torch


def estimate_model_grad_for(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss: torch.Tensor,
    device: torch.device,
    nes_num_iters: int = 200,
    nes_sigma: float = 1 / 255,
) -> torch.Tensor:
    # NES algorithm for gradient approximation
    with torch.no_grad():
        grad: torch.Tensor = torch.zeros(x.shape).to(device)
        for i in range(nes_num_iters):
            u_i = torch.normal(
                mean=torch.zeros(grad.shape), std=torch.ones(grad.shape)
            ).to(device)

            yhat = model(x + nes_sigma * u_i)
            p_y = (
                yhat.take_along_dim(y.reshape(-1, 1), dim=1)
                .reshape(-1, 1, 1, 1)
                .to(device)
            )
            grad += p_y * u_i

            yhat = model(x - nes_sigma * u_i)
            p_y = (
                yhat.take_along_dim(y.reshape(-1, 1), dim=1)
                .reshape(-1, 1, 1, 1)
                .to(device)
            )
            grad -= p_y * u_i
        grad /= 2 * nes_num_iters * nes_sigma
    return grad
