import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaseResNet18(nn.Module):
    def __init__(self):
        super(BaseResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)

    def forward(self, x):
        return self.resnet(x)

######################################################
# TODO: either define the Activation Shaping Module as a nn.Module
#class ActivationShapingModule(nn.Module):
#...
#
# OR as a function that shall be hooked via 'register_forward_hook'
#def activation_shaping_hook(module, input, output):
#...
#
def asm_hook_generator(M) :
    def activation_shaping_hook(module, input, output) :        
        # Binarization of M (if it is not already binarized)
        bin_values = (M == 0) + (M == 1)
        if not torch.all(bin_values) :
            M[M <= 0] = 0
            M[M > 0] = 1

        # Binarization of last layer output
        output[output <= 0] = 0
        output[output > 0] = 1

        bin_prod = M * output

        return bin_prod 
    return activation_shaping_hook


######################################################
class ASHResNet18(nn.Module):
    def __init__(self):
        super(ASHResNet18, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)
        self.hooks = dict()
   
    def put_asm_after_layer(self, layer, asm_hook) :
        model_layer = None
        for name, module in self.resnet.named_children() :
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

######################################################
