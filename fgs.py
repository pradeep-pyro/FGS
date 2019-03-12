import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fgs(model, input_image, label, targeted=False, alpha=0.02, iterations=1,
        reg=1e-2, clamp=((-2.118, -2.036, -1.804), (2.249, 2.429, 2.64)),
        use_cuda=True):
    """
    Fast gradient sign method for generating adverserial attack examples

    Parameters
    ----------
    model: torch.nn.Module
        PyTorch pretrained model with weights already loaded
    input_image: PyTorch 4D Tensor
        Initial image that is to be modified for adverserial attack
    label: int
        True label of the input_image if untargeted attack
        (adverserial image will be classified into this label)
        OR
        Desired label of input_image if targeted attack
        (adverserial image will not be classified into this label)
    targeted: bool
        Whether targeted or untargeted attack (default: False)
    alpha: float
        Step size for updating image with sign of gradient (default: 0.02)
    iterations: int
        Number of iterations to repeat the algorithm (default: 1)
    reg: float
        MSE regularization to keep adverserial and original image close
        (default: 1e-2)
    clamp: tuple
        Min and max values for the image used to clamp image into valid range
        after each iteration (default: based on ImageNet range).
        Set to (None, None) to avoid clamping
    use_cuda: bool
        Try to use CUDA if available
    """
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.to(device)
    model.eval()
    crit = nn.CrossEntropyLoss().to(device)
    input_image = input_image.to(device)
    img_var = input_image.clone().requires_grad_(True).to(device)
    label_var = torch.LongTensor([label]).to(device)
    for _ in range(iterations):
        img_var.grad = None
        out = model(img_var)
        # Cross entropy + MSE regularization between adverserial and original image
        loss = crit(out, label_var) + reg * F.mse_loss(img_var, input_image)
        loss.backward()
        noise = alpha * torch.sign(img_var.grad.data)
        if targeted:
            img_var.data = img_var.data - noise
        else:
            img_var.data = img_var.data + noise
        # Clamp image into valid range
        if clamp[0] is not None and clamp[1] is not None:
            assert len(clamp[0]) == len(clamp[1])
            for ch in range(len(clamp[0])):
                img_var.data[:, ch, :, :].clamp_(clamp[0][ch], clamp[1][ch])
    return img_var.cpu().detach()
