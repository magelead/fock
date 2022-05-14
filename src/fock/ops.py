
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



def displacement_matrix(alpha, D):
    """creates the single mode displacement matrix"""
    batch_size = alpha.shape[0]

    idxs = [(j, k) for j in range(D) for k in range(j)] # [(1, 0), (2, 0), (2, 1)] for D=3
    idxs = batchify_indices(idxs, batch_size)
    idxs = torch.tensor(idxs).T


    signs = [(-1.) ** (j-k) 
             for j in range(D) for k in range(j)]
    signs = signs * batch_size  
    signs = torch.tensor(signs)
    # tensor([-1, 1, -1,    -1, 1, -1])



    values = [alpha ** (j-k) * np.sqrt(binom(j, k) / factorial(j-k)) 
              for j in range(D) for k in range(j)]
    values = torch.stack(values, dim=-1)  
    values = values.reshape(-1)
    # tensor([1.0000+0.j, 0.7071+0.j, 1.4142+0.j,     2.0000+0.j, 2.8284+0.j, 2.8284+0.j])



    dense_shape = [batch_size, D, D]
        
    eye_diag = torch.stack([torch.eye(D)]*batch_size) # (batch_size, D, D)
    
    sign_lower_diag = torch.zeros(dense_shape).index_put_([dim for dim in idxs], signs)
    sign_matrix = sign_lower_diag + eye_diag

    lower_diag = torch.zeros(dense_shape, dtype=torch.complex64).index_put_([dim for dim in idxs], values)
    E = lower_diag + eye_diag
    

    E_prime = torch.conj(E) * sign_matrix
    

    eqn = 'aik,ajk->aij' 
  
    prefactor = torch.unsqueeze(torch.unsqueeze(torch.exp(-0.5 * torch.abs(alpha) ** 2).to(torch.complex64), -1), -1)
    
    D_alpha = prefactor * torch.einsum(eqn, E, E_prime)
    
    return D_alpha


def batchify_indices(idxs, batch_size):
    """adds batch indices to the index numbering"""
    return [(bdx,) + idxs[i] for bdx in range(batch_size) for i in range(len(idxs))]



def displacement(alpha, mode, in_modes, D, pure=True):
    """returns displacement unitary matrix on specified input modes"""
    matrix = displacement_matrix(alpha, D)
    output = single_mode_gate(matrix, mode, in_modes, pure)

    return output

