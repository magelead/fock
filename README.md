# Fock



A continuous variable quantum neural network simulating framework written in PyTorch. 

The name of the package comes from the fact that the Fock basis representation is used as the underlining mathematics.



# How to develop/test



## Create an env with pytorch and tensorflow





## Install `fock` from repo in editible mode

```
git clone https://github.com/magelead/fock
cd fock
# upgrade pip & run following cmd in the same directory as `pyproject.toml`
pip install -e . 
```

## Migrate strawberryfields==0.10.0 to tf2.x

Download `strawberryfields==0.10.0` to `tests/strawberryfields` and modify code if necessary

## Develop new function

## Write test code then run 


```
cd tests/
python test_ops.py
```





# Push to Github

in the same directory as `pyproject.toml`

```
git add .
git commit -m ''
git push
```





# Publish to PyPI

in the same directory as `pyproject.toml`

```
python -m pip install --upgrade build
```

```
python -m build
```

```
python -m pip install --upgrade twine
```

```
python -m twine upload dist/*
```


# Read the Docs


Built with Sphinx using a theme provided by Read the Docs.


# Todo


* test_circuit.py should use MAE, now there is no absolute fucntion
* BS
* PS
* S
* Kerr


# References

https://packaging.python.org/en/latest/tutorials/packaging-projects/

https://stackoverflow.com/questions/5341006/where-should-i-put-tests-when-packaging-python-modules

[strawberryfields==0.10.0](https://pypi.org/project/StrawberryFields/0.10.0/) 