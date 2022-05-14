from itertools import product
import numbers
from string import ascii_lowercase as indices

import numpy as np
import tensorflow as tf
import torch
from scipy.special import binom, factorial

import fock
import strawberryfields  as sf



def test_circuit_reset():

    cir_sf = sf.Circuit(num_modes=2, cutoff_dim=3, hbar=2., pure=False, batch_size=3)
    cir_fock = fock.Circuit(num_modes=2, cutoff_dim=3, hbar=2., pure=False, batch_size=3)
    error = (cir_sf._state.numpy() - cir_fock._state.numpy()).mean()

    return error.real + error.imag


# todo random args
def test_circuit_displacemnet(pure=True, scalar=True):
    

    if scalar:

        r=torch.randn((), requires_grad=True)
        phi=torch.randn((),requires_grad=True)
        alpha = r * torch.exp(1j * phi)
        alpha = tf.Variable(alpha.detach())

    else:

        r=torch.randn(3, requires_grad=True)                # (batch_size, )
        phi=torch.randn(3, requires_grad=True)
        alpha = r * torch.exp(1j * phi)
        alpha = tf.Variable(alpha.detach())




    cir_sf = sf.Circuit(num_modes=2, cutoff_dim=3, hbar=2., pure=pure, batch_size=3)
    cir_sf.displacement(alpha=alpha, mode=0)
    #print('debug cir_sf._state', cir_sf._state)

    cir_fock = fock.Circuit(num_modes=2, cutoff_dim=3, hbar=2., pure=pure, batch_size=3)
    cir_fock.displacement(r=r, phi=phi, mode=0)
    #print('debug cir_fock._state', cir_fock._state)

    error = (cir_sf._state.numpy() - cir_fock._state.detach().numpy()).mean()

    return error.real + error.imag




if __name__ == '__main__':

    


    # test `Circuit.reset`
    for i in range(10):
        error = test_circuit_reset()
        if error > 1.e-6:
            raise ValueError(f'test `Circuit.reset` failed! error={error}')


    # test `Circuit.displacement`
    for i in range(10):
        error = test_circuit_displacemnet(pure=True, scalar=True)
        if error > 1.e-6:
            raise ValueError(f'test `Circuit.displacement` failed! error={error}')


    # test `Circuit.displacement`
    for i in range(10):
        error = test_circuit_displacemnet(pure=True, scalar=False)
        if error > 1.e-6:
            raise ValueError(f'test `Circuit.displacement` failed! error={error}')


    # test `Circuit.displacement`
    for i in range(10):
        error = test_circuit_displacemnet(pure=False, scalar=True)
        if error > 1.e-6:
            raise ValueError(f'test `Circuit.displacement` failed! error={error}')


    # test `Circuit.displacement`
    for i in range(10):
        error = test_circuit_displacemnet(pure=False, scalar=False)
        if error > 1.e-6:
            raise ValueError(f'test `Circuit.displacement` failed! error={error}')



    print('Passed!')










