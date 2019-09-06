# Split Learning

## Installation
The code uses PySyft framework, which needs to be installed and slightly modified in order to be able to use GPUs. PySyft requires Python >= 3.6.  To install the right version, run the following commands: 
```
$ git clone https://github.com/OpenMined/PySyft.git 
$ cd PySyft
$ pip install -r requirements.txt
$ cd .. 
$ mv native.py PySyft/syft/frameworks/torch/tensors/interpreters/ 
$ cd PySyft
$ python3 setup.py install 
```


## Running experiments
To run the experiments, run the command: 
```
$ python3 experience_name num_workers
```
where num_workers is the number of clients/data owners we want in the experiment. 
This script will therefore first create num_workers clients, split into num_workers parts the dataset (CIFAR if "_cifar" extension, else MNIST). These parts are distributed to the clients. 
Then the model is trained according to the selected method:
fedl: Vanilla Federated Learning
favg: Federated Averaging
SL: Split Learning

It prints out the time it took to train the model, and the memory exchanged by each worker. 
