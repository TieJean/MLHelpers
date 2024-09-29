import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU, linear_layer=nn.Linear):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.linear_layer = linear_layer
        self.input_dim = layer_sizes[0]
        self.output_dim = layer_sizes[-1]
        self.build(layer_sizes)

    def build(self, layer_sizes):
        for i in range(len(layer_sizes) - 1):
            self.layers.append(self.linear_layer(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation after the last layer
                self.layers.append(self.activation())

    def forward(self, x):
        assert x.shape[-1] == self.input_dim
        for layer in self.layers:
            x = layer(x)
        return x
