import torch
from torchvision import transforms


def preprocess(im, width=224, height=224, mean=(0.485, 0.456, 0.406),
               std=(0.229, 0.224, 0.225)):
    """
    Convert an RGB image into a 4D PyTorch tensor
    Resizes and performs standard normalization

    Parameters:
    -----------
    im: PIL image
        Input image
    width: int
        Required width (default: 224)
    height: int
        Required height (default: 224)
    mean: float or tuple of floats
        Mean of the pixels
        (default: [0.485, 0.456, 0.406] from ImageNet)
    std: float or tuple of floats
        Standard deviation of the pixels
        (default: [0.229, 0.224, 0.225] from ImageNet)
    """
    tforms = transforms.Compose([transforms.Resize((width, height)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std),
                                 transforms.Lambda(lambda x : torch.unsqueeze(x, 0))])
    return tforms(im)


def postprocess(im, mean=[-0.485, -0.456, -0.406],
                std=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225]):
    """
    Convert 4D PyTorch tensor into an RGB image

    Parameters:
    -----------
    im: 4D PyTorch Tensor
        Input image
    mean: float or tuple of floats
        Negative of mean of the pixels that was used for preprocessing
        (default: [-0.485, -0.456, -0.406] from ImageNet)
    std: float or tuple of floats
        Inverse of standard deviation of the pixels that was used for preprocessing
        (default: [1/0.229, 1/0.224, 1/0.225] from ImageNet)
    """
    tforms = transforms.Compose([transforms.Lambda(lambda x : torch.squeeze(x, 0)),
                                 transforms.Normalize([0.0, 0.0, 0.0], std),
                                 transforms.Normalize(mean, [1.0, 1.0, 1.0]),
                                 transforms.ToPILImage()])
    return tforms(im)
