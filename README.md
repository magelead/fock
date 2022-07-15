# Fock



A continuous variable quantum neural network simulating framework written in PyTorch. 

The name of the package comes from the fact that the Fock basis representation is used as the underlining mathematics.

















# How to develop/test



## Create an env named fock with following packages

```
numpy==1.22.3
tensorflow-macos==2.8.0
torch==1.11.0
```


## Install `fock` from repo in editible mode
in the same directory as `setup.cfg`
```
git clone https://github.com/magelead/fock
cd fock
pip install -e .
```

## Download `strawberryfields==0.10.0` to `tests/strawberryfields` and Migrate strawberryfields==0.10.0 to tf2.x

## Develop new function

## Write test code then run 


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