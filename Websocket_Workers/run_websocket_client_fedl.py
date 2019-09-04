import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms
import logging
import argparse
import sys
import time

import syft as sy
from syft.workers import WebsocketClientWorker
from syft.workers import VirtualWorker
from syft.frameworks.torch.federated import utils

logger = logging.getLogger(__name__)

LOG_INTERVAL = 25

class Arguments():
    def __init__(self, no_cuda):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 2
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = no_cuda
        self.seed = 1
        self.log_interval = 30
        self.save_model = False
        self.use_virtual = False
        self.verbose = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.max_pool2d(x, 2, 2)
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return f.log_softmax(x, dim=1)



def model_size(model):
    size = 0
    for p in model.parameters():
        size += p.element_size()*p.nelement()
    return size

def train(args, model, device, federated_train_loader, optimizer, epoch, clients_mem):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader): # <-- now it is a distributed dataset
        model.send(data.location) # <-- NEW: send the model to the right location
        data, target = data.to(device), target.to(device)
        #i = int(data.location.id.split()[-1])
        if data.location.id == 'alice':
            i = 0
        elif data.location.id == 'bob':
            i = 1
        else :
            i = 2

        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.get() # <-- NEW: get the model back
        clients_mem[i] += 2*model_size(model)
        if batch_idx % args.log_interval == 0:
            loss = loss.get() # <-- NEW: get the loss back
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
                100. * batch_idx / len(federated_train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += f.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))




def main():
    args = Arguments(False)

    hook = sy.TorchHook(torch)

    if args.use_virtual:
        alice = VirtualWorker(id="alice", hook=hook, verbose=args.verbose)
        bob = VirtualWorker(id="bob", hook=hook, verbose=args.verbose)
        charlie = VirtualWorker(id="charlie", hook=hook, verbose=args.verbose)
    else:
        kwargs_websocket = {"host": "localhost", "hook": hook, "verbose": args.verbose}
        alice = WebsocketClientWorker(id="alice", port=8777, **kwargs_websocket)
        bob = WebsocketClientWorker(id="bob", port=8778, **kwargs_websocket)
        charlie = WebsocketClientWorker(id="charlie", port=8779, **kwargs_websocket)

    workers = [alice, bob, charlie]
    clients_mem = torch.zeros(len(workers))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
    # TODO Quickhack. Actually need to fix the problem moving the model to CUDA\n",
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    kwargs = {"num_workers": 0, "pin_memory": False} if use_cuda else {}

    federated_train_loader = sy.FederatedDataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ).federate(tuple(workers)),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs,
    )

    start = time.time()
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, federated_train_loader, optimizer, epoch, clients_mem)
        test(args, model, device, test_loader)
        t = time.time()
        print(t-start)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    end = time.time()
    print(end-start)
    print("Memory exchanged : ",clients_mem)

if __name__ == "__main__":
    FORMAT = "%(asctime)s %(levelname)s %(filename)s(l:%(lineno)d) - %(message)s"
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(format=FORMAT, level=LOG_LEVEL)

    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.DEBUG)
    websockets_logger.addHandler(logging.StreamHandler())

    main()
