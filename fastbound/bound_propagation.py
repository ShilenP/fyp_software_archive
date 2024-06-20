import torch
from torch import nn

from fastbound.bound_propagation_gru import ibp_propagate_gru
from fastbound.bound_propagation_linear import ibp_propagate_linear

def ibp_propagation(lower_bound: torch.Tensor, upper_bound: torch.Tensor, layer, hidden_lower=None, hidden_upper=None):
    """
    Propagate bounds through a neural network layer
    Args:
        lower_bound: Lower bound of the input
        upper_bound: Upper bound of the input
        layer: Neural network layer
        hidden_lower: Lower bound of the hidden state (required for GRU layers)
        hidden_upper: Upper bound of the hidden state (required for GRU layers)
    Returns:
        new_lower_bound: Lower bound of the output
        new_upper_bound: Upper bound of the output
        *hidden_lower: Lower bound of the hidden state (only for GRU layers)
        *hidden_upper: Upper bound of the hidden state (only for GRU layers)
    """
    if isinstance(layer, nn.Flatten):
        lower_bound = layer(lower_bound)
        upper_bound = layer(upper_bound)
    elif isinstance(layer, nn.Linear):
        lower_bound, upper_bound = ibp_propagate_linear(lower_bound, upper_bound, layer.weight, layer.bias)
    elif isinstance(layer, nn.ReLU):
        lower_bound, upper_bound = lower_bound.relu(), upper_bound.relu()
    elif isinstance(layer, nn.Sigmoid):
        lower_bound, upper_bound = lower_bound.sigmoid(), upper_bound.sigmoid()
    elif isinstance(layer, nn.GRU):
        if hidden_lower is None or hidden_upper is None:
            raise ValueError("Hidden bounds are required for GRU layers")
        return ibp_propagate_gru(lower_bound, upper_bound, hidden_lower, hidden_upper, layer)
    elif isinstance(layer, nn.Sequential):
        for sub_layer in layer:
            lower_bound, upper_bound = ibp_propagation(lower_bound, upper_bound, sub_layer)
    elif isinstance(layer, nn.ReplicationPad1d):
        lower_bound = layer(lower_bound)
        upper_bound = layer(upper_bound)
    elif isinstance(layer, nn.GELU):
        gelu1 = layer(lower_bound)
        gelu2 = layer(upper_bound)
        lower_bound = torch.min(gelu1, gelu2)
        upper_bound = torch.max(gelu1, gelu2)
    elif isinstance(layer, nn.Dropout):
        pass
    else:
        raise ValueError(f"Layer {layer} not supported")
    return lower_bound, upper_bound