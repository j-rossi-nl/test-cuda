# test-cuda

Test run using Bert. Trains a text classifier from a small subset of IMDB Reviews (128 samples in train / 128 in test).

## Install Poetry
* `poetry` is the dependency manager. [website](https://python-poetry.org)
* Install it on the default Python environment with `pip install poetry`

## Create Virtual Env
```shell
poetry install
poetry run poe torchXXX-cudaYYY
```

Available:
* Torch 1.9.0
  * CUDA 10.2 `torch190-cuda102`
  * CUDA 11.1 `torch190-cuda111`
* Torch 1.7.1
  * CUDA 9.2 `torch171-cuda92`
  * CUDA 10.1 `torch171-cuda101`
  * CUDA 10.2 `torch171-cuda102`
  * CUDA 11.0 `torch171-cuda110`

# Run
```shell
poetry run python bert.py
```

# Destroy Virtual Env
Get ready for the next run by destroying the virtual env

_(using FISH shell)_
```shell
for e in (poetry env list)                                                                                                                              0 (1s) < 10:03
    set sp (string split " " $e)
    poetry env remove $sp[1]
end
```