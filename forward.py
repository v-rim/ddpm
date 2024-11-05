import torch
from utils import tensor_to_PIL, extract


def q_sample(x_start, t, noise=None, constants=None):
    """Samples from the forward process in a diffusion model.

    Args:
        x_start (torch.Tensor): The initial data tensor.
        t (torch.Tensor): The timestep to sample at.
        noise (torch.Tensor): The noise tensor to use for sampling. If None, generates a new noise tensor.
        constants (DiffusionConstants): The constants for the diffusion process.

    Returns:
        torch.Tensor: The sampled tensor at timestep `t`.
    """
    sqrt_alphas_cumprod = constants.sqrt_alphas_cumprod
    sqrt_one_minus_alphas_cumprod = constants.sqrt_one_minus_alphas_cumprod

    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x_start, t, constants):
    """Generate a noisy image.

    Applies the diffusion process to generate a noisy image from a starting image `x_start` after `t` diffusion steps.

    Args:
        x_start (torch.Tensor): The starting image tensor.
        t (int): The number of diffusion steps to apply.
        constants (DiffusionConstants): The constants for the diffusion process.

    Returns:
        PIL.Image.Image: The noisy image.
    """
    x_noisy = q_sample(x_start=x_start, t=t, constants=constants)
    noisy_image = tensor_to_PIL(x_noisy)
    return noisy_image
