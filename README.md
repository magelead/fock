# Fock

A continuous variable quantum neural network simulating framework written in PyTorch. 

The name of the package comes from the fact that the Fock basis representation is used as the  munderlining mathematics.

It is forked from strawberryfields==0.10.0.



# Development Notes (Don't look)

torch.einsum('i,j->ij', x, y)

$|x\rangle \otimes |y \rangle = x_i y_j$ 

torch.einsum: Multiply each component of each operand then sum if possible.

Tensor product genralizes outer product.




