# Chameleon

[![built with Python3.6](https://img.shields.io/badge/build%20with-python%203.6-red.svg)](https://www.python.org/)
[![built with PyTorch1.4](https://img.shields.io/badge/build%20with-pytorch%201.4-brightgreen.svg)](https://pytorch.org/)

### Introduction

In this repository you will find a pytorch implementation of Chameleon with
 the short-term and long-term stores. 

### Project Structure
The project is structured as follows:

- [`models/`](models): In this folder the main MobileNetV1 model is defined
 along with the custom Batch Renormalization Pytorch layer.
- [`chameleon.py`](chameleon.py): Main Chameleon algorithm.
- [`data_loader.py`](data_loader.py): CORe50 data loader.
- [`params.cfg`](params.cfg): Hyperparameters that will be used in the main
 experiment on CORe50-NI.
- [`README.md`](README.md): This instructions file.
- [`utils.py`](utils.py): Utility functions used in the rest of the code.

### Getting Started

When using anaconda virtual environment all you need to do is run the following 
command and conda will install everything for you. 
See [environment.yml](./environment.yml):

    conda env create --file environment.yml
    conda activate chameleon-env
    
Then to reproduce the results on the CORe50-NI benchmark you just
 need to run the following code:
 
 ```bash
python chameleon.py --run 3 --scenario ni
```
