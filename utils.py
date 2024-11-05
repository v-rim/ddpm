import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.v2 import Compose


def PIL_to_tensor(examples, image_column="image", resize=None):
    """Mapping function to convert dataset of PIL images to tensors.

    Applies the following transformations in order:
    - ToTensor (converts to [0, 1])
    - Normalize to [-1, 1]
    - Resize to the specified size (if provided)

    Intended to be used with datasets.Dataset.map()
    
    Args:
        examples (datasets.Dataset): Dataset of examples
        image_column (str): The column in the dataset containing the PIL images
        resize (int): The size to resize the images to. If None, does nothing
    """
    transform = (
        Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 2) - 1),
                transforms.Resize(resize),
            ]
        )
        if resize is not None
        else Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x * 2) - 1),
            ]
        )
    )

    examples["pixel_values"] = [
        transform(image.convert("L")) for image in examples[image_column]
    ]

    return examples


def tensor_to_PIL(example):
    """Converts a tensor to a PIL Image.

     Applies the following transformations in order:
    - Map values from [-1, 1] to [0, 1]
    - ToPILImage

    Args:
        example (torch.Tensor): The input tensor to be converted.

    Returns:
        PIL.Image.Image: The resulting PIL Image.
    """
    transform = Compose(
        [
            transforms.Lambda(lambda x: (x + 1) / 2),
            transforms.ToPILImage(),
        ]
    )

    return transform(example)


def extract(a, t, x_shape):
    """Get specified indices from a source matrix and reshape them.

    Extracts values from tensor `a` at indices specified by tensor `t` and reshapes the result. `a` will usually be a tensor of constants at every timestep and `t` will be the actual timesteps to get. `x_shape` is needed to make sure dimensions line up.

    Args:
        a (torch.Tensor): The source tensor from which values are extracted.
        t (torch.Tensor): A tensor containing indices to gather from `a`.
        x_shape (torch.Size): The shape of the tensor `x` which determines the reshaping of the output.

    Returns:
        torch.Tensor: A tensor with values gathered from `a` at indices specified by `t`, reshaped to match `x_shape`.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionConstants:
    """A class used to cache constants for the diffusion process.

    Attributes:
        timesteps (int): The number of timesteps to generate constants for.
        betas (torch.Tensor): The beta value for each timestep.
        alphas (torch.Tensor): The alpha value for each timestep.
        sqrt_recip_alphas (torch.Tensor): The square root of the reciprocal of the alpha value for each timestep.
        sqrt_alphas_cumprod (torch.Tensor): The square root of the cumulative product of the alpha values.
        sqrt_one_minus_alphas_cumprod (torch.Tensor): The square root of the cumulative product of 1 - alpha values.
        posterior_variance (torch.Tensor): The posterior variance for each timestep.
    """

    def __init__(self, timesteps, s=0.008):
        self.timesteps = timesteps
        # self.betas = self.cosine_beta_schedule(timesteps=timesteps, s=s)
        self.betas = self.linear_beta_schedule(timesteps=timesteps)

        # define alphas
        self.alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Generates beta values using a cosine schedule.

        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672

        Args:
            timesteps (int): The number of timesteps to generate beta values for.
            s (float): The cosine schedule parameter.

        Returns:
            torch.Tensor: The generated beta values.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def linear_beta_schedule(self, timesteps):
        """Generates beta values using a linear schedule.
        
        Args:
            timesteps (int): The number of timesteps to generate beta values for.
            
        Returns:
            torch.Tensor: The generated beta values.
        """
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr