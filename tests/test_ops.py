
import torch
import numpy as np

import fock
import strawberryfields  as sf


def test_displacement_matrix():
   
    alpha = torch.randn(3, dtype=torch.complex64)   # (batch_size, )
    matrix = fock.ops.displacement_matrix(alpha, D=10).numpy()

    alpha_sf = alpha
    matrix_sf = sf.ops.displacement_matrix(alpha_sf, D=10, batched=True).numpy()

    error = np.mean(np.abs((matrix-matrix_sf)))

    return error, alpha.numpy()





if __name__ == '__main__':

    # test `displacement_matrix`
    for i in range(100):
        error, alpha = test_displacement_matrix()
        if error > 5e-6:
            raise ValueError(f'test `displacement_matrix` failed! error={error} alpha={alpha}')
       

    print('Passed!')



