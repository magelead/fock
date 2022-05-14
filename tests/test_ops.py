from string import ascii_lowercase as indices

import numpy as np
import tensorflow as tf
import torch
from scipy.special import binom, factorial

import fock
import strawberryfields  as sf


def test_displacement_matrix():
   
    alpha = torch.randn(3, requires_grad=True, dtype=torch.complex64)   # (batch_size, )
    matrix = fock.ops.displacement_matrix(alpha, D=10).detach().numpy()

    alpha_sf = tf.Variable(alpha.detach())
    matrix_sf = sf.ops.displacement_matrix(alpha_sf, D=10, batched=True).numpy()

    error = (matrix-matrix_sf).mean()

    return error.real + error.imag, alpha.detach().numpy()





if __name__ == '__main__':

    # test `displacement_matrix`
    for i in range(10):
        error, alpha = test_displacement_matrix()
        if error > 1.e-6:
            raise ValueError(f'test `displacement_matrix` failed! error={error} alpha={alpha}')
       

    print('Passed!')



