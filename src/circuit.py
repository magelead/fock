
import numbers

import numpy as np
import torch

import ops



class Circuit:
    r"""Base class for representing and operating on a collection of
    CV quantum optics modes in the Fock basis.
    The modes are initialized in the (multimode) vacuum state,
    using the Fock representation with given cutoff_dim.
    The state of the modes is manipulated by calling the various methods.
    
    `self._batched=True` all the time
    `self._state_is_pure` -> `self._pure`
    """
    def __init__(self, num_modes, cutoff_dim, hbar=2., pure=True, batch_size=None):
        self._batch_size = batch_size
        self.reset(pure, num_modes=num_modes, cutoff_dim=cutoff_dim, hbar=hbar)

    # todo stress test
    def displacement(self, r, phi, mode):
        """
        Apply the displacement operator to the specified mode.
        Parameters:
            r (Tensor) -  displacement magnitude, shape (batch_size, )
            phi (Tensor) - displacement angle, shape (batch_size, )
            mode (int) - the mode to apply the displacement operator
        """
        alpha = r * torch.exp(1j * phi)
        new_state = ops.displacement(alpha, mode, self._state, self._cutoff_dim, self._pure)
        self._update_state(new_state)


    def _valid_modes(self, modes):
        """modes (list[int] or non-negative int): The mode(s) of the CV circuit."""
        if isinstance(modes, int):
            modes = [modes]

        for mode in modes:
            if mode < 0:
                raise ValueError("Specified mode number(s) cannot be negative.")
            elif mode >= self._num_modes:
                raise ValueError("Specified mode number(s) are not compatible with number of modes.")

        if len(modes) != len(set(modes)):
            raise ValueError("The specified modes cannot appear multiple times.")

        return True
    
    def _check_incompatible_batches(self, *params):
        """Helper function for verifying that all the params have the same batch size."""   
        for idx, p in enumerate(params):
            param_batch_size = p.shape[0]
            if idx == 0:
                ref_batch_size = param_batch_size
            else:
                if param_batch_size != ref_batch_size:
                    raise ValueError("Parameters have incompatible batch sizes.")


    def _make_vac_states(self, cutoff_dim):
        """Make vacuum state tensors"""
        v = torch.zeros(cutoff_dim, dtype=torch.complex64)
        v[0] = 1.+0.j
        self._single_mode_pure_vac = v
        self._single_mode_mixed_vac = torch.einsum('i,j->ij', v, v)
        self._single_mode_pure_vac = torch.stack([self._single_mode_pure_vac] * self._batch_size)
        self._single_mode_mixed_vac = torch.stack([self._single_mode_mixed_vac] * self._batch_size)

    def _update_state(self, new_state):
        """Helper function to update the state history and the current state"""
        self._state_history.append(new_state)
        self._state = new_state
    
    def reset(self, pure=True, num_modes=None, cutoff_dim=None, hbar=None):
        r"""
        Resets the state of the circuit to have all modes in vacuum.
        For all the parameters, None means unchanged.

        Args:
            pure (bool): if True, the reset circuit will represent its state as a pure state. If False, the representation will be mixed.
            num_modes (int): sets the number of modes in the reset circuit.
            cutoff_dim (int): new Fock space cutoff dimension to use.
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

        if cutoff_dim is not None:
            if not isinstance(cutoff_dim, int) or cutoff_dim < 1:
                raise ValueError("Argument 'cutoff_dim' must be a positive integer")
            self._cutoff_dim = cutoff_dim

        if hbar is not None:
            if not isinstance(hbar, numbers.Real) or hbar <= 0:
                raise ValueError("Argument 'hbar' must be a positive number")
            self._hbar = hbar

        self._state_history = []
        
        self._make_vac_states(self._cutoff_dim)
        single_mode_vac = self._single_mode_pure_vac if pure else self._single_mode_mixed_vac
        
        if self._num_modes == 1:
            vac = single_mode_vac
        else: 
            vac = ops.combine_single_modes([single_mode_vac] * self._num_modes)
        
        self._update_state(vac)



if __name__ == '__main__':

    cir = Circuit(num_modes=2, cutoff_dim=3, hbar=2., pure=False, batch_size=3)

    cir.displacement(r=torch.tensor([1., 2., 3.], requires_grad=True), phi=torch.tensor([0., 0., 0.], requires_grad=True), mode=0)

    cir._state





    