import torch
from torch import nn
from fastbound.bound_propagation_linear import ibp_propagate_linear
from fastbound.tools import min_max_hadamard_product

def ibp_propagate_gru_cell(x_lower_i: torch.Tensor, x_upper_i: torch.Tensor, hidden_lower: torch.Tensor, hidden_upper: torch.Tensor, weight_ih: torch.Tensor, weight_hh: torch.Tensor, bias_ih: torch.Tensor, bias_hh: torch.Tensor):
    """
    Propagate bounds through a GRU cell
    Args:
        x_lower_i: Lower bound of the input at a specific time step
        x_upper_i: Upper bound of the input at a specific time step
        hidden_lower: Lower bound of the hidden state
        hidden_upper: Upper bound of the hidden state
        weight_ih: Weight matrix of the input-hidden connections
        weight_hh: Weight matrix of the hidden-hidden connections
        bias_ih: Bias vector of the input-hidden connections
        bias_hh: Bias vector of the hidden-hidden connections
    Returns:
        h_next_lower: Lower bound of the hidden state at the next time step
        h_next_upper: Upper bound of the hidden state at the next time step
    """
    W_ir, W_iz, W_in = weight_ih.chunk(3, 0)
    W_hr, W_hz, W_hn = weight_hh.chunk(3, 0)
    b_ir, b_iz, b_in = bias_ih.chunk(3, 0)
    b_hr, b_hz, b_hn = bias_hh.chunk(3, 0)

    r1_lower, r1_upper = ibp_propagate_linear(x_lower_i, x_upper_i, W_ir, b_ir)
    r2_lower, r2_upper = ibp_propagate_linear(hidden_lower, hidden_upper, W_hr, b_hr)
    r_lower = torch.sigmoid(r1_lower + r2_lower)
    r_upper = torch.sigmoid(r1_upper + r2_upper)

    z1_lower, z1_upper = ibp_propagate_linear(x_lower_i, x_upper_i, W_iz, b_iz)
    z2_lower, z2_upper = ibp_propagate_linear(hidden_lower, hidden_upper, W_hz, b_hz)
    z_lower = torch.sigmoid(z1_lower + z2_lower)
    z_upper = torch.sigmoid(z1_upper + z2_upper)

    n1_lower, n1_upper = ibp_propagate_linear(x_lower_i, x_upper_i, W_in, b_in)
    n2_lower, n2_upper = ibp_propagate_linear(hidden_lower, hidden_upper, W_hn, b_hn)
    n3_lower, n3_upper = min_max_hadamard_product(r_lower, r_upper, n2_lower, n2_upper)
    n_lower, n_upper = torch.tanh(n1_lower + n3_lower), torch.tanh(n1_upper + n3_upper)

    h1_next_lower, h1_next_upper = min_max_hadamard_product(1-z_lower, 1-z_upper, n_lower, n_upper)
    h2_next_lower, h2_next_upper = min_max_hadamard_product(z_lower, z_upper, hidden_lower, hidden_upper)
    h_next_lower, h_next_upper = h1_next_lower + h2_next_lower, h1_next_upper + h2_next_upper
    return h_next_lower, h_next_upper

def ibp_propagate_gru(x_lower: torch.Tensor, x_upper: torch.Tensor, hidden_lower: torch.Tensor, hidden_upper: torch.Tensor, gru: nn.GRU):
    """
    Propagate bounds through a GRU layer
    Args:
        x_lower: Lower bound of the input
        x_upper: Upper bound of the input
        hidden_lower: Lower bound of the hidden state
        hidden_upper: Upper bound of the hidden state
        gru: GRU layer
    Returns:
        output_lower: Lower bound of the output
        output_upper: Upper bound of the output
        hidden_lower: Lower bound of the hidden state
        hidden_upper: Upper bound of the hidden state
    """
    steps = x_lower.size(1)
    output_lower = []
    output_upper = []
    for i in range(steps):
        x_lower_i, x_upper_i = x_lower[:, i], x_upper[:, i]
        hidden_lower, hidden_upper = ibp_propagate_gru_cell(x_lower_i, x_upper_i, hidden_lower, hidden_upper, gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0)
        output_lower.append(hidden_lower)
        output_upper.append(hidden_upper)

    output_lower = torch.stack(output_lower, dim=1) 
    output_upper = torch.stack(output_upper, dim=1)
    return output_lower, output_upper, hidden_lower, hidden_upper
