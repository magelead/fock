from string import ascii_lowercase as indices

import numpy as np
import tensorflow as tf
import torch

import ops



def displacement_matrix_tf(alpha, D, batched=False):
    """creates the single mode displacement matrix"""
    if batched:
        batch_size = alpha.shape[0]
    alpha = tf.cast(alpha, def_type)
    idxs = [(j, k) for j in range(D) for k in range(j)]
    values = [alpha ** (j-k) * tf.cast(tf.sqrt(binom(j, k) / factorial(j-k)), def_type)
              for j in range(D) for k in range(j)]
    values = tf.stack(values, axis=-1)
    dense_shape = [D, D]
    vals = [1.0] * D
    ind = [(idx, idx) for idx in range(D)]
    if batched:
        dense_shape = [batch_size] + dense_shape
        vals = vals * batch_size
        ind = batchify_indices(ind, batch_size)
    eye_diag = tf.SparseTensor(ind, vals, dense_shape)
    signs = [(-1) ** (j-k) for j in range(D) for k in range(j)]
    if batched:
        idxs = batchify_indices(idxs, batch_size)
        signs = signs * batch_size
    sign_lower_diag = tf.cast(tf.SparseTensor(idxs, signs, dense_shape), tf.float32)
    sign_matrix = tf.compat.v1.sparse_add(eye_diag, sign_lower_diag)
    sign_matrix = tf.cast(tf.compat.v1.sparse_tensor_to_dense(sign_matrix), def_type)
    lower_diag = tf.scatter_nd(idxs, tf.reshape(values, [-1]), dense_shape)
    E = tf.cast(tf.eye(D), def_type) + lower_diag
    E_prime = tf.compat.v1.conj(E) * sign_matrix
    if batched:
        eqn = 'aik,ajk->aij' # pylint: disable=bad-whitespace
    else:
        eqn = 'ik,jk->ij' # pylint: disable=bad-whitespace
    prefactor = tf.expand_dims(tf.expand_dims(tf.cast(tf.exp(-0.5 * tf.abs(alpha) ** 2), def_type), -1), -1)
    D_alpha = prefactor * tf.einsum(eqn, E, E_prime)
    return D_alpha



def test_displacement_matrix():
    print('test_displacement_matrix:')
    alpha = torch.randn(3, requires_grad=True, dtype=torch.complex64)   # (batch_size, )
    matrix = ops.displacement_matrix(alpha, D=10).detach().numpy()

    alpha_tf = tf.Variable(alpha.detach())
    matrix_tf = displacement_matrix_tf(alpha_tf, D=10, batched=True).numpy()

    error = (matrix-matrix_tf).mean()
    return error.real + error.imag, alpha.detach().numpy()





if __name__ == '__main__':


    for i in range(10):
      error, alpha = test_displacement_matrix()
      if error > 1.e-8:
        print(error, alpha)



