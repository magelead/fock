# https://arxiv.org/pdf/2004.11002.pdf



from string import ascii_lowercase as indices
max_num_indices = len(indices)

import numpy as np
import torch
from scipy.special import binom, factorial



def combine_single_modes(modes_list):
    """Group together a list of single modes (each having ndim=1 or ndim=2) into a composite mode system."""
    batch_offset = 1
    num_modes = len(modes_list)
    
    if num_modes <= 1:
        raise ValueError("'modes_list' must have at least two modes")

    ndims = np.array([mode.ndim - batch_offset for mode in modes_list])
    if min(ndims) < 1 or max(ndims) > 2:
        raise ValueError("Each mode in 'modes_list' can only have ndim=1 or ndim=2")

    if np.all(ndims == 1):
        # All modes are represented as pure states.
        # Can return combined state also as pure state.
        # basic form (no batch):
        # 'a,b,c,...,x,y,z->abc...xyz' 
        max_num = max_num_indices - batch_offset
        if num_modes > max_num:
            raise NotImplementedError("The max number of supported modes for this operation with pure states is currently {}".format(max_num))
        batch_index = indices[:batch_offset]                         # 'a'
        out_str = indices[batch_offset : batch_offset + num_modes]   # 'bcde'
        modes_str = ",".join([batch_index + idx for idx in out_str]) # 'ab,ac,ad,ae'
        eqn = "{}->{}".format(modes_str, batch_index + out_str)      # 'ab,ac,ad,ae->abcde'
        einsum_inputs = modes_list
    else:
        # Return combined state as mixed states.
        # basic form:
        # e.g., if first mode is pure and second is mixed...
        # 'ab,cd,...->abcd...'
        # where ab will belong to the first mode (density matrix)
        # and cd will belong to the second mode (density matrix)
        max_num = (max_num_indices - batch_offset) // 2
        if num_modes > max_num:
            raise NotImplementedError("The max number of supported modes for this operation with mixed states is currently {}".format(max_num))
        batch_index = indices[:batch_offset]
        mode_idxs = [indices[slice(batch_offset + idx, batch_offset + idx + 2)] for idx in range(0, 2 * num_modes, 2)] # each mode gets a pair of consecutive indices
        # mode_idxs = ['bc', 'de']
        eqn_rhs = batch_index + "".join(mode_idxs) # 'abcde'
        eqn_idxs = [batch_index + m for m in mode_idxs] # ['abc', 'ade'] 
        eqn_lhs = ",".join(eqn_idxs) #  'abc, ade'
        eqn = eqn_lhs + "->" + eqn_rhs
        einsum_inputs = modes_list
    
    combined_modes = torch.einsum(eqn, *einsum_inputs)
    return combined_modes



def single_mode_gate(matrix, mode, in_modes, pure=True):
    """basic form:
    'ab,cde...b...xyz->cde...a...xyz' (pure state)      U|Ψ>
    'ab,ef...bc...xyz,cd->ef...ad...xyz' (mixed state)  UρU†
    """

    batch_offset = 1
 
    batch_index = indices[:batch_offset]
    left_gate_str = indices[batch_offset : batch_offset + 2] # |a><b|
    num_indices = len(in_modes.shape)
    if pure:
        num_modes = num_indices - batch_offset
        mode_size = 1
    else:
        right_gate_str = indices[batch_offset + 2 : batch_offset + 4] # |c><d|
        num_modes = (num_indices - batch_offset) // 2
        mode_size = 2
    max_len = len(indices) - 2 * mode_size - batch_offset #26 letters 2*mode_size reserved for gates, 1 for batch
    if num_modes == 0:
        raise ValueError("'in_modes' must have at least one mode")
    if num_modes > max_len:
        raise NotImplementedError("The max number of supported modes for this operation is currently {}".format(max_len))
    if mode < 0 or mode >= num_modes:
        raise ValueError("'mode' argument is not compatible with number of in_modes")
    else:
        other_modes_indices = indices[batch_offset + 2 * mode_size : batch_offset + (1 + num_modes) * mode_size]
        if pure:
            eqn_lhs = "{},{}{}{}{}".format(batch_index + left_gate_str, batch_index, other_modes_indices[:mode * mode_size], left_gate_str[1], other_modes_indices[mode * mode_size:])
            eqn_rhs = "".join([batch_index, other_modes_indices[:mode * mode_size], left_gate_str[0], other_modes_indices[mode * mode_size:]])
        else:
            eqn_lhs = "{},{}{}{}{}{},{}".format(batch_index + left_gate_str, batch_index, other_modes_indices[:mode * mode_size], left_gate_str[1], right_gate_str[0], other_modes_indices[mode * mode_size:], batch_index + right_gate_str)
            eqn_rhs = "".join([batch_index, other_modes_indices[:mode * mode_size], left_gate_str[0], right_gate_str[1], other_modes_indices[mode * mode_size:]])

    eqn = eqn_lhs + "->" + eqn_rhs
    
    einsum_inputs = [matrix, in_modes]
    if not pure:
        transposed_axis = [0, 2, 1]
        einsum_inputs.append(torch.permute(torch.conj(matrix), transposed_axis))
    print('debug', eqn)
    output = torch.einsum(eqn, *einsum_inputs)
    return output


#---------------------------------------------------------------------------------------------------------------------------------------


def displacement_matrix(r, phi, cutoff, dtype):  # pragma: no cover
    r"""Calculates the matrix elements of the displacement gate using a recurrence relation.
    Args:
        r (tensor): batched displacement magnitude shape = (batch_size,)
        phi (tensor): batched displacement angle shape = (batch_size,)
        cutoff (int): Fock ladder cutoff
        dtype (data type): Specifies the data type used for the calculation
    Returns:
        tensor[complex]: matrix representing the displacement operation.
    """

    r = r.to(dtype)
    phi = phi.to(dtype)
    batch_size = r.shape[0]


    D = torch.zeros((batch_size, cutoff, cutoff)).to(dtype)
    sqrt = torch.sqrt(torch.arange(cutoff)).to(dtype)
    
    alpha0 = r * torch.exp(1j * phi)
    alpha1 = -r * torch.exp(-1j * phi)


    D[:, 0, 0] = torch.exp(-0.5 * r**2)
    
    for m in range(1, cutoff):

        D[:, m, 0] = alpha0 / sqrt[m] * D[:, m - 1, 0].clone()

    

    for n in range(1, cutoff):
    	D[:, 0, n] = alpha1 / sqrt[n] * D[:, 0, n - 1].clone()


    for m in range(1, cutoff):
        for n in range(1, cutoff):
            D[:, m, n] = alpha1 / sqrt[n] * D[:, m, n - 1].clone() + sqrt[m] / sqrt[n] * D[:, m - 1, n - 1].clone()
    
    return D







def phase_shifter_matrix(phi, cutoff, dtype):
    """creates the single mode phase shifter matrix
    Args:
        phi (tensor): batched angle shape = (batch_size,)
    """
    phi = phi.to(dtype)
   
    diag = [torch.exp(1j * phi * n) for n in range(cutoff)]
    diag = torch.stack(diag, dim=1)
    diag_matrix = torch.diag_embed(diag)
    return diag_matrix


def kerr_interaction_matrix(kappa, cutoff, dtype):
    """creates the single mode Kerr interaction matrix
    Args:
        kappa (tensor): batched angle shape = (batch_size,)
    """
    kappa = kappa.to(dtype)

    diag = [torch.exp(1j * kappa * n ** 2) for n in range(cutoff)]
    diag = torch.stack(diag, dim=1)
    diag_matrix = torch.diag_embed(diag)
    return diag_matrix


#---------------------------------------------------------------------------------------------------------------------------------------

def displacement(r, phi, mode, in_modes, cutoff, pure=True, dtype=torch.complex64):
    """returns displacement unitary matrix applied on specified input modes"""
    matrix = displacement_matrix(r, phi, cutoff, dtype)
    output = single_mode_gate(matrix, mode, in_modes, pure)

    return output




def phase_shifter(phi, mode, in_modes, cutoff, pure=True, dtype=torch.complex64):
    """returns phase shift unitary matrix on specified input modes"""
    matrix = phase_shifter_matrix(phi, cutoff, dtype=dtype)
    output = single_mode_gate(matrix, mode, in_modes, pure)
    return output


def kerr_interaction(kappa, mode, in_modes, cutoff, pure=True, dtype=torch.complex64):
    """returns Kerr unitary matrix on specified input modes"""
    matrix = kerr_interaction_matrix(kappa, cutoff, dtype=dtype)
    output = single_mode_gate(matrix, mode, in_modes, pure)
    return output








