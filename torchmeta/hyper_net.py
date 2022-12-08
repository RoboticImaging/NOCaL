"""

This code was originally use in the light field network.
See here: https://github.com/vsitzmann/light-field-networks

Network to take the latent space from previous network and convert
to weights for implicit render model.
Created: 25/02/22
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from torchmeta import custom_layers


class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        """

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        """
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = custom_layers.FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                                       num_hidden_layers=hyper_hidden_layers, hidden_ch=hyper_hidden_features,
                                       outermost_linear=True, norm='layernorm')
            if 'weight' in name:
                hn.net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1], siren=True))
            elif 'bias' in name:
                hn.net[-1].apply(lambda m: hyper_bias_init(m, siren=True))

            self.nets.append(hn)

    def forward(self, z):
        """
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        """
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


############################
# Initialization scheme
def hyper_weight_init(m, in_features_main_net, siren=False):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1e1



def hyper_bias_init(m, siren=False):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e1
