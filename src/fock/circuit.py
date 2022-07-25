
import numbers

import numpy as np
import torch

from . import ops



class Circuit:
    r"""Base class for representing and operating on a collection of
    CV quantum optics modes in the Fock basis.
    The modes are initialized in the (multimode) vacuum state,
    using the Fock representation with given cutoff.
    The state of the modes is manipulated by calling the various methods.
    
    `self._batched=True` all the time
    `self._state_is_pure` -> `self._pure`
    """
    def __init__(self, num_modes, cutoff, hbar=2., pure=True, batch_size=None, dtype=torch.complex64):
        self._batch_size = batch_size
        self.reset(pure, num_modes=num_modes, cutoff=cutoff, hbar=hbar, dtype=dtype)
        print('Init a CV circuit...')


    def displacement(self, r, phi, mode):
        """
        Apply the displacement operator to the specified mode.
        Parameters:
            r (Tensor) -  displacement magnitude >0, shape=(batch_size, ) or scalar
            phi (Tensor) - displacement angle, shape (batch_size, ) or scalar
            mode (int) - the mode to apply the displacement operator

        Notes:
            scalar parameter meaning different circuits share the same parameter across a batch
            scalar will be broadcast to vector of batch_size
            r phi should be float32 for complex64, float64 for complex128
        """

        # broadcast the scalar to a vector 
        if r.ndim == 0:
            r = torch.stack([r] * self._batch_size)
        if phi.ndim == 0:
            phi = torch.stack([phi] * self._batch_size)

        new_state = ops.displacement(r, phi, mode, self._state, self._cutoff, self._pure, self._dtype)
        self._update_state(new_state)


    def phase_shifter(self, phi, mode):
        """
        Apply the phase shifter (rotation) operator to the specified mode.
        Parameters:
            phi (Tensor) - phase shift angle, shape=(batch_size, ) or scalar
            mode (int) - the mode to apply the displacement operator
        """

        if phi.ndim == 0:
            phi = torch.stack([phi] * self._batch_size)

        new_state = ops.phase_shifter(phi, mode, self._state, self._cutoff, self._pure, self._dtype)
        self._update_state(new_state)


    def kerr_interaction(self, kappa, mode):
        """
        Apply the Kerr interaction operator to the specified mode.
        """

        if kappa.ndim == 0:
            kappa = torch.stack([kappa] * self._batch_size)

        new_state = ops.kerr_interaction(kappa, mode, self._state, self._cutoff, self._pure, self._dtype)
        self._update_state(new_state)



    def _make_vac_states(self, cutoff):
        """Make vacuum state tensors"""
        v = torch.zeros(cutoff, dtype=self._dtype)
        v[0] = 1.+0.j
        self._single_mode_pure_vac = v
        self._single_mode_mixed_vac = torch.einsum('i,j->ij', v, v)
        self._single_mode_pure_vac = torch.stack([self._single_mode_pure_vac] * self._batch_size)
        self._single_mode_mixed_vac = torch.stack([self._single_mode_mixed_vac] * self._batch_size)

    def _update_state(self, new_state):
        """Helper function to update the state history and the current state"""
        self._state_history.append(new_state)
        self._state = new_state
    
    def reset(self, pure=True, num_modes=None, cutoff=None, hbar=None, dtype=torch.complex64):
        r"""
        Resets the state of the circuit to have all modes in vacuum.
        For all the parameters, None means unchanged.

        Args:
            pure (bool): if True, the reset circuit will represent its state as a pure state. If False, the representation will be mixed.
            num_modes (int): sets the number of modes in the reset circuit.
            cutoff (int): new Fock space cutoff dimension to use.
            hbar (float): new :math:`\hbar` value.
        """
        if pure is not None:
            if not isinstance(pure, bool):
                raise ValueError("Argument 'pure' must be either True or False")
            self._pure = pure

        if num_modes is not None:
            if not isinstance(num_modes, int):
                raise ValueError("Argument 'num_subsystems' must be a positive integer")
            self._num_modes = num_modes

        if cutoff is not None:
            if not isinstance(cutoff, int) or cutoff < 1:
                raise ValueError("Argument 'cutoff' must be a positive integer")
            self._cutoff = cutoff

        if hbar is not None:
            if not isinstance(hbar, numbers.Real) or hbar <= 0:
                raise ValueError("Argument 'hbar' must be a positive number")
            self._hbar = hbar

        if dtype not in (torch.complex64, torch.complex128):
            raise ValueError("Argument 'dtype' must be a complex PyTorch data type")
        self._dtype = dtype

        self._state_history = []
        
        self._make_vac_states(self._cutoff)
        single_mode_vac = self._single_mode_pure_vac if pure else self._single_mode_mixed_vac
        
        if self._num_modes == 1:
            vac = single_mode_vac
        else: 
            vac = ops.combine_single_modes([single_mode_vac] * self._num_modes)
        
        self._update_state(vac)









    