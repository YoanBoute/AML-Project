import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from copy import copy

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)


def asm_hook_generator(M) :
    def activation_shaping_hook(module, input, output) :
        # Binarization of M (if it is not already binarized)
        bin_values = (M == 0) + (M == 1)
        if not torch.all(bin_values) :
            M[M <= 0] = 0
            M[M > 0] = 1

        # Binarization of last layer output
        bin_output = torch.clone(output)
        bin_output[bin_output <= 0] = 0
        bin_output[bin_output > 0] = 1

        bin_prod = M * bin_output

        return bin_prod
    return activation_shaping_hook


def asm_hook_generator_no_binarization(M):
    """extension 2 - part 1 (no binarization)"""
    def activation_shaping_hook(module, input, output) :
        return M * output
    return activation_shaping_hook


def asm_hook_generator_top_k(M, K) :
    """extension 2 - part 2 (top-k binarization)"""
    def activation_shaping_hook(module, input, output) :
        # Binarization of M (if it is not already binarized)
        bin_values = (M == 0) + (M == 1)
        if not torch.all(bin_values):
            M[M <= 0] = 0
            M[M > 0] = 1

        # compute top K elements of A
        A = torch.clone(output)
        _, indices = torch.topk(A, K)
        # Create a mask with zeros everywhere except the top k indices
        mask = torch.zeros_like(A)
        mask[indices] = 1

        return M * mask * A
    return activation_shaping_hook



class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = dict()

    def put_asm_after_layer(self, layer, asm_hook) :
        model_layer = None
        for name, module in self.resnet.named_modules() :
            if name == layer :
                model_layer = module
                break
        if model_layer is None :
            raise BaseException(f"Error : The layer {layer} couldn't be found in the model")

        hook = model_layer.register_forward_hook(asm_hook)
        self.hooks[layer] = hook

    def remove_asm_after_layer(self, layer) :
        if self.hooks.get(layer) is not None and self.hooks[layer] is not None :
            self.hooks[layer].remove()
            self.hooks[layer] = None
        else :
            raise BaseException("Error : no hook attached to this layer")

    def forward(self, x):
       return self.resnet(x)

