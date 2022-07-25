import numpy as np
import tensorflow as tf
import torch

import fock
import strawberryfields.backends.tfbackend  as sf

def test_circuit_reset():

    cir_sf = sf.circuit.Circuit(num_modes=2, cutoff_dim=3, pure=False, batch_size=3, dtype=tf.complex128)

    cir_fock = fock.Circuit(num_modes=2, cutoff=3, hbar=2., pure=False, batch_size=3, dtype=torch.complex128)

    error = np.mean(np.abs((cir_sf._state.numpy() - cir_fock._state.numpy())))

    return error


def test_circuit_displacemnet(pure=True, scalar=True, mode=0):
    

    if scalar:

        r=torch.rand(())
        phi=torch.randn(())

        r_sf = r.numpy()
        phi_sf = phi.numpy()
      
    else:

        r=torch.rand(3)                # (batch_size, )
        phi=torch.randn(3)
        
        r_sf = r.numpy()
        phi_sf = phi.numpy()


    cir_sf = sf.circuit.Circuit(num_modes=2, cutoff_dim=3, pure=pure, batch_size=3, dtype=tf.complex128)
    cir_sf.displacement(r=r_sf, phi=phi_sf, mode=mode)
    #print('debug cir_sf._state', cir_sf._state)

    cir_fock = fock.Circuit(num_modes=2, cutoff=3, hbar=2., pure=pure, batch_size=3, dtype=torch.complex128)
    cir_fock.displacement(r=r, phi=phi, mode=mode)
    #print('debug cir_fock._state', cir_fock._state)

    error = np.mean(np.abs(cir_sf._state.numpy() - cir_fock._state.numpy()))

   
    return error



def test_circuit_grad_displacemnet(pure):
    r=torch.rand(3, requires_grad=True)                # (batch_size, )
    phi=torch.randn(3, requires_grad=True)  
    
    cir_fock = fock.Circuit(num_modes=2, cutoff=3, hbar=2., pure=pure, batch_size=3, dtype=torch.complex128)
    cir_fock.displacement(r=r, phi=phi, mode=1)
    cir_fock._state.sum().backward()
    r_grad  = r.grad
    phi_grad = phi.grad

        
    r_sf = tf.Variable(r.detach().numpy())
    phi_sf = tf.Variable(phi.detach().numpy())

    with tf.GradientTape() as tape:
        cir_sf = sf.circuit.Circuit(num_modes=2, cutoff_dim=3, pure=pure, batch_size=3, dtype=tf.complex128)
        cir_sf.displacement(r=r_sf, phi=phi_sf, mode=1)
        sum_sf =  tf.reduce_sum(cir_sf._state)

    [r_grad_sf, phi_grad_sf] = tape.gradient(sum_sf, [r_sf, phi_sf])

    error = np.mean(np.abs((r_grad - r_grad_sf))) + np.mean(np.abs((phi_grad - phi_grad_sf)))

    return error



def test_circuit_phase_shifter(pure=True, scalar=True, mode=0):
    

    if scalar:

        phi=torch.randn(())

        phi_sf = phi.numpy()
      
    else:

        # (batch_size, )
        phi=torch.randn(3)
        
        phi_sf = phi.numpy()


    cir_sf = sf.circuit.Circuit(num_modes=2, cutoff_dim=3, pure=pure, batch_size=3, dtype=tf.complex128)
    cir_sf.phase_shift(phi_sf, mode)


    cir_fock = fock.Circuit(num_modes=2, cutoff=3, hbar=2., pure=pure, batch_size=3, dtype=torch.complex128)
    cir_fock.phase_shifter(phi, mode)

    error = np.mean(np.abs(cir_sf._state.numpy() - cir_fock._state.numpy()))

   
    return error


if __name__ == '__main__':

    


    # test `Circuit.reset`
    for i in range(10):
        error = test_circuit_reset()
        if error > 1e-6:
            raise ValueError(f'test `Circuit.reset` failed! error={error}')







    # test `Circuit.displacement`
    for i in range(10):
        error = test_circuit_displacemnet(pure=True, scalar=True)
        if error > 1e-6:
            raise ValueError(f'test `Circuit.displacement` failed! error={error}')


    # test `Circuit.displacement`
    for i in range(10):
        error = test_circuit_displacemnet(pure=True, scalar=False)
        if error > 1e-6:
            raise ValueError(f'test `Circuit.displacement` failed! error={error}')


    # test `Circuit.displacement`
    for i in range(10):
        error = test_circuit_displacemnet(pure=False, scalar=True)
        if error > 1e-6:
            raise ValueError(f'test `Circuit.displacement` failed! error={error}')


    # test `Circuit.displacement`
    for i in range(10):
        error = test_circuit_displacemnet(pure=False, scalar=False)
        if error > 1e-6:
            raise ValueError(f'test `Circuit.displacement` failed! error={error}')


    # test `grad of Circuit.displacement`
    for i in range(10):
        error = test_circuit_grad_displacemnet(pure=True)
        if error > 1e-6:
            raise ValueError(f'test `grad of Circuit.displacement` failed! error={error}')


    # test `grad of Circuit.displacement`
    for i in range(10):
        error = test_circuit_grad_displacemnet(pure=False)
        if error > 1e-6:
            raise ValueError(f'test `grad of Circuit.displacement` failed! error={error}')














    # test `Circuit.phase_shifter`   
    for i in range(10):
        error = test_circuit_phase_shifter(pure=True, scalar=True)
        if error > 1e-6:
            raise ValueError(f'test `Circuit.phase_shifter` failed! error={error}')


    # test `Circuit.phase_shifter`
    for i in range(10):
        error = test_circuit_phase_shifter(pure=True, scalar=False)
        if error > 1e-6:
            raise ValueError(f'test `Circuit.phase_shifter` failed! error={error}')


    # test `Circuit.phase_shifter`
    for i in range(10):
        error = test_circuit_phase_shifter(pure=False, scalar=True)
        if error > 1e-6:
            raise ValueError(f'test `Circuit.phase_shifter` failed! error={error}')

    
    # test `Circuit.phase_shifter`
    for i in range(10):
        error = test_circuit_phase_shifter(pure=False, scalar=False)
        if error > 1e-6:
            raise ValueError(f'test `Circuit.phase_shifter` failed! error={error}')




    print('\nPassed!\n')



    






