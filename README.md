# model_quantizing_demo
a simple demo for model quantizing, based on distiller, only keeping the quantizing part, and supported to latest pytorch.

## steps
* **Install Packages**
  1. install virtual environment
  - python3 -m venv venv
  - cd model_quantizing_demo
  - pip3 install -r requirements.txt
  2. install packages
  - pip3 install -e .
* **Train A Model**
  1. cd examples
  2. python3 train_model.py --arch resnet32_cifar ../../data.cifar10 -p 30 -j=1 --lr=0.01
* **Quantize A Model**
  1. move the checkpoints to trained_models folder
  2. python3 quantize_model.py --arch resnet32_cifar ../../data.cifar10 --resume ../trained_models/(checkpoint_name).pth.tar --quantize-eval --evaluate

## supported version
  - can running correctly on Python 3.6.9 / Ubuntu 18.04 