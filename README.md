# TAO Toolkit - Core

* [Overview](#Overview)
* [Getting Started](#Getting Started)

## <a name='Overview'></a>Overview
TAO-Core is a Python package hosted on the NVIDIA Python Package Index. It comprises of modules containing core packages for TAO Toolkit DNN containers. 

## <a name='Getting Started'></a>Getting Started
TAO-Core needs to be re-compiled depending upon the Python version required. To build the wheel, lauch the base container with the required python version and execute:
```sh
nvidia-docker run -it -v `pwd`:/tao-core <BASE_CONTAINER>
bash release/python/build_wheel.sh
```

## <a name='License'></a>License
This project is licensed under the [Apache-2.0](./LICENSE) License.