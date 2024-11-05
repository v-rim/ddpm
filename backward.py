import torch
import torch.nn.functional as F
from forward import q_sample
from utils import extract
from tqdm import tqdm


def p_losses(denoise_model, x_start, t, noise=None, constants=None):
    """Computes the loss between the predicted noise and the actual noise at a given time step.
    
   Args:
        denoise_model (nn.Module): The model used to predict the noise.
        x_start (torch.Tensor): A batch of source images. Dimensions are (batch_size, channels, height, width).
        t (torch.Tensor): A batch of time steps. Dimensions are (batch_size, 1).
        noise (torch.Tensor): The noise at the given time step. If None, random noise is in the shape of x_start used.
        constants (Namespace): The constants used in the reverse diffusion process.
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise, constants=constants)
    predicted_noise = denoise_model(x_noisy, t)
    return F.smooth_l1_loss(noise, predicted_noise)


@torch.no_grad()
def p_sample(model, x, t, t_index, constants):
    """One step of the reverse diffusion process.
    
    Sample from the reverse diffusion process at a given time step.
    
    Args:
        model (nn.Module): The model used to predict the noise.
        x (torch.Tensor): The image to sample from. Dimensions are (batch_size, channels, height, width).
        t (torch.Tensor): The time step to sample from. Dimensions are (batch_size, 1).
        t_index (int): The index of the time step.
        constants (Namespace): The constants used in the reverse diffusion process.
    """
    betas_t = extract(constants.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        constants.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(constants.sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(constants.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape, constants):
    """Apply the entire reverse diffusion process.
    
    Args:
        model (nn.Module): The model used to predict the noise.
        shape (tuple): The shape of image batches. Dimensions are (batch_size, channels, height, width).
        constants (Namespace): The constants used in the reverse diffusion process.
        
    Returns:
        list: A list of image tensors in the given shape
    """
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(
        reversed(range(0, constants.timesteps)),
        desc="sampling loop time step",
        total=constants.timesteps,
    ):
        img = p_sample(
            model,
            img,
            torch.full((b,), i, device=device, dtype=torch.long),
            i,
            constants,
        )
        imgs.append(img.cpu())
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3, constants=None):
    return p_sample_loop(
        model, shape=(batch_size, channels, image_size, image_size), constants=constants
    )


@torch.no_grad()
def p_refine(img, steps, model, shape, constants):
    """Repeatedly apply the last step of reverse diffusion to clean up minor artifacts."""
    device = next(model.parameters()).device

    b = shape[0]
    img = img.to(device)

    for i in tqdm(range(steps), desc="refine loop time step"):
        img = p_sample(
            model,
            img,
            torch.full((b,), 0, device=device, dtype=torch.long),
            0,
            constants,
        )
    return img.cpu()
