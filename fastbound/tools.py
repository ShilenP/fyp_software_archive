import torch

def min_max_hadamard_product(x_lower: torch.Tensor, x_upper: torch.Tensor, y_lower: torch.Tensor, y_upper: torch.Tensor):
    """
    Compute the elementwise minimum and maximum of the Hadamard product of two tensors
    Args:
        x_lower: Lower bound of the first tensor
        x_upper: Upper bound of the first tensor
        y_lower: Lower bound of the second tensor
        y_upper: Upper bound of the second tensor
    Returns:
        elementwise_min: Elementwise minimum of the Hadamard product
        elementwise_max: Elementwise maximum of the Hadamard product
    """
    prod1 = x_lower * y_lower
    prod2 = x_lower * y_upper
    prod3 = x_upper * y_lower
    prod4 = x_upper * y_upper
    elementwise_min = torch.min(torch.min(prod1, prod2), torch.min(prod3, prod4))
    elementwise_max = torch.max(torch.max(prod1, prod2), torch.max(prod3, prod4))
    return elementwise_min, elementwise_max
