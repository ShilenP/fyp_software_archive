import torch

def ibp_propagate_linear(lower_bound: torch.Tensor, upper_bound: torch.Tensor, weight: torch.Tensor, bias=None):
    """
    Propagate bounds through a linear layer
    Args:
        lower_bound: Lower bound of the input
        upper_bound: Upper bound of the input
        weight: Weight matrix of the linear layer
        bias: Bias vector of the linear layer
    Returns:
        new_lower_bound: Lower bound of the output
        new_upper_bound: Upper bound of the output
    """
    weight_positive = torch.clamp(weight, min=0)
    weight_negative = torch.clamp(weight, max=0)
    new_lower_bound = lower_bound @ weight_positive.t() + upper_bound @ weight_negative.t() + (0 if bias is None else bias)
    new_upper_bound = upper_bound @ weight_positive.t() + lower_bound @ weight_negative.t() + (0 if bias is None else bias)
    return new_lower_bound, new_upper_bound
