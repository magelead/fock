
import torch
import tensorflow as tf
import numpy as np

import fock
import strawberryfields.backends.tfbackend  as sf




def test_displacement_matrix(cutoff, dtype):
   
    if dtype == 'complex64':
        torch_dtype = torch.complex64
        tf_dtype = tf.complex64
    
    if dtype == 'complex128':
        torch_dtype = torch.complex128
        tf_dtype = tf.complex128
 
    r = 10 * torch.rand(2)    # batch_size = 2
    phi = 10 * torch.randn(2)
    matrix = fock.ops.displacement_matrix(r, phi, cutoff, dtype=torch_dtype).numpy()

    r_sf = r.numpy()
    phi_sf = phi.numpy()
    matrix_sf = sf.ops.displacement_matrix(r_sf, phi_sf, cutoff, batched=True, dtype=tf_dtype).numpy()

    error = np.mean(np.abs((matrix-matrix_sf)))


 
    return error, r, phi


def test_grad_displacement_matrix(cutoff, dtype):

    if dtype == 'complex64':
        torch_dtype = torch.complex64
        tf_dtype = tf.complex64
    
    if dtype == 'complex128':
        torch_dtype = torch.complex128
        tf_dtype = tf.complex128
 
    r = torch.rand(2, requires_grad=True)    # batch_size = 2
    phi = torch.randn(2, requires_grad=True)
    matrix = fock.ops.displacement_matrix(r, phi, cutoff, dtype=torch_dtype)
    matrix.sum().backward()
    r_grad  = r.grad
    phi_grad = phi.grad


    
    r_sf = tf.Variable(r.detach().numpy())
    phi_sf = tf.Variable(phi.detach().numpy())
    with tf.GradientTape() as tape:
        matrix_sf = sf.ops.displacement_matrix(r_sf, phi_sf, cutoff, batched=True, dtype=tf_dtype)
        sum_sf = tf.reduce_sum(matrix_sf)

    [r_grad_sf, phi_grad_sf] = tape.gradient(sum_sf, [r_sf, phi_sf])


    error = np.mean(np.abs((r_grad - r_grad_sf))) + np.mean(np.abs((phi_grad - phi_grad_sf)))
    return error, r, phi



 


def test_phase_shifter_matrix(cutoff, dtype):
   
    if dtype == 'complex64':
        torch_dtype = torch.complex64
        tf_dtype = tf.complex64
    
    if dtype == 'complex128':
        torch_dtype = torch.complex128
        tf_dtype = tf.complex128
 
     
    phi = 10 * torch.randn(2) # batch_size = 2
    matrix = fock.ops.phase_shifter_matrix(phi, cutoff, dtype=torch_dtype).numpy()

    phi_sf = phi.numpy()
    matrix_sf = sf.ops.phase_shifter_matrix(phi_sf, cutoff, batched=True, dtype=tf_dtype).numpy()

    error = np.mean(np.abs((matrix-matrix_sf)))


 
    return error, phi






if __name__ == '__main__':


    # test `displacement_matrix`
    for i in range(100):
        error, r, phi = test_displacement_matrix(cutoff=10, dtype='complex128')
        if error > 1e-5:
            raise ValueError(f' `test_displacement_matrix` failed! error={error} r={r} phi={phi}')
       

    # test `grad_displacement_matrix`
    for i in range(100):
        error, r, phi = test_grad_displacement_matrix(cutoff=10, dtype='complex128')
        if error > 1e-5:
            raise ValueError(f' `test_grad_displacement_matrix` failed! error={error} r={r} phi={phi}')


    # test `phase_shifter_matrix`
    for i in range(100):
        error, r = test_phase_shifter_matrix(cutoff=10, dtype='complex128')
        if error > 1e-5:
            raise ValueError(f' `test_phase_shifter_matrix` failed! error={error} phi={phi}')


    print('\nPassed!\n')

