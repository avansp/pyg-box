<div align="center">

# PYG-BOX

[![python](https://img.shields.io/badge/-Python_3.12-blue?logo=python&logoColor=white)]()
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/docs/pytorch/stable/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![pyg](https://img.shields.io/badge/PyG_2.7.0+-coral?logo=pyg)](https://pytorch-geometric.readthedocs.io)

</div>

A testing sandbox for graph-based neural network from [PyTorch Geometry](https://pytorch-geometric.readthedocs.io) library.

## Installation

```{bash}
pip install -r requirements.txt
```

Install `torch-scatter` and `torch-sparse` libraries using specific torch CUDA version. Follow these steps: 

* Check your torch version by using this command:
    ```
    python -c "import torch; print(torch.__version__)"
    ```
    for example
    ```text
    2.9.1+cu128
    ```

* Then install with that specific CUDA version of your torch library
    ```
    pip install torch-sparse torch-scatter -f "https://data.pyg.org/whl/torch-{CUDA_VERSION}.html"
    ```

    For the example above:
    ```
    pip install torch-sparse torch-scatter -f "https://data.pyg.org/whl/torch-2.9.1+cu128.html"
    ```

Install the package script (note use `-e` for development)
```{bash}
pip install .
```

## Graph classification with MUTAG dataset

Notebook to learn: [graph_classification.ipynb](notebooks/graph_classification.ipynb)

Training:
```bash
train task=mutag
```


## Node classification with CORA dataset

Notebook to learn: [cora_clustering.ipynb](notebooks/cora_clustering.ipynb)

Training:
```bash
train task=cora
```
