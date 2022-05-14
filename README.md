# Fock

A continuous variable quantum neural network simulating framework written in PyTorch. 

The name of the package comes from the fact that the Fock basis representation is used as the  munderlining mathematics.

It is forked from strawberryfields==0.10.0.





# Development Notes (Don't look)

torch.einsum('i,j->ij', x, y)

$|x\rangle \otimes |y \rangle = x_i y_j$ 

torch.einsum: Multiply each component of each operand then sum if possible.

Tensor product genralizes outer product.



# Todo

test_circuit.py













# How to debug







**test env named torch**



numpy==1.22.3

tensorflow-macos==2.8.0

torch==1.11.0



install fock from repo in editible mode

in the same directory as `setup.cfg`

```
pip install -e .
```





**migrate**

migrate strawberryfields==0.10.0 to tf2.0 then run comparison test to fock



**test**

write test logic

```
 cd tests/
 python3 test_ops.py
```





# Push to Github

in the same directory as `setup.cfg`

```
git add .
git commit -m ''
git push
```





# Publish to PyPI

in the same directory as `setup.cfg`

```
python3 -m pip install --upgrade build
```

```
python3 -m build
```

```
python3 -m pip install --upgrade twine
```

```
python3 -m twine upload dist/*
```







# References

https://packaging.python.org/en/latest/tutorials/packaging-projects/

https://stackoverflow.com/questions/5341006/where-should-i-put-tests-when-packaging-python-modules

