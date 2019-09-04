import sys
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
from syft.workers import WebsocketClientWorker
from syft.workers import VirtualWorker
from syft.frameworks.torch.federated import utils

num_workers = int(sys.argv[-1])

class Arguments():
    def __init__(self, no_cuda):
        self.batch_size = 256
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = no_cuda
        self.seed = 1
        self.log_interval = 25
        self.save_model = False
        self.federate_after_n_batches = 50


# Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def model_size(model):
    size = 0
    for p in model.parameters():
        size += p.element_size()*p.nelement()
    return size

def train_on_batches(worker, batches, model_in, device, lr,epoch,clients_mem,args):
    """Train the model on the worker on the provided batches
    Args:
        worker(syft.workers.BaseWorker): worker on which the
        training will be executed
        batches: batches of data of this worker
        model_in: machine learning model, training will be done on a copy
        device (torch.device): where to run the training
        lr: learning rate of the training steps
    Returns:
        model, loss: obtained model and loss after training
    """
    model = model_in.copy()
    optimizer = optim.SGD(model.parameters(), lr=lr)  # TODO momentum is not supported at the moment

    i = int(worker.id.split()[-1])
    model.train()
    tmp = model_size(model)
    clients_mem[i] += tmp
    model.send(worker)


    loss_local = False
    #D = {"alice":0,"bob":1,"charlie":2}

    for batch_idx, (data, target) in enumerate(batches):
        loss_local = False
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            loss = loss.get()  # <-- NEW: get the loss back
            loss_local = True
            print(
                "Epoch {} Train Worker {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    worker.id,
                    batch_idx,
                    len(batches),
                    100.0 * batch_idx / len(batches),
                    loss.item(),
                )
            )

    if not loss_local:
        loss = loss.get()  # <-- NEW: get the loss back
    model.get()  # <-- NEW: get the model back
    clients_mem[i] += model_size(model)
    return model, loss


def get_next_batches(fdataloader: sy.FederatedDataLoader, nr_batches: int):
    """retrieve next nr_batches of the federated data loader and group
    the batches by worker
    Args:
        fdataloader (sy.FederatedDataLoader): federated data loader
        over which the function will iterate
        nr_batches (int): number of batches (per worker) to retrieve
    Returns:
        Dict[syft.workers.BaseWorker, List[batches]]
    """
    batches = {}
    #print("workers id:",fdataloader.workers)
    for worker_id in fdataloader.workers:
        worker = fdataloader.federated_dataset.datasets[worker_id].location
        #print("worker",worker)
        batches[worker] = []
    try:
        for i in range(nr_batches):
            next_batches = next(fdataloader)
            for worker in next_batches:
                batches[worker].append(next_batches[worker])
    except StopIteration:
        pass
    return batches


def train(args,model, device, federated_train_loader, lr, federate_after_n_batches,epoch,clients_mem):
    model.train()

    nr_batches = federate_after_n_batches

    models = {}
    loss_values = {}

    iter(federated_train_loader)  # initialize iterators
    batches = get_next_batches(federated_train_loader, nr_batches)
    counter = 0

    while True:
        print(
            "Starting training round, batches [{}, {}]".format(counter, counter + nr_batches)
        )
        data_for_all_workers = True
        for worker in batches:
            curr_batches = batches[worker]
            if curr_batches:
                models[worker], loss_values[worker] = train_on_batches(
                    worker, curr_batches, model, device, lr,epoch,clients_mem,args
                )
            else:
                data_for_all_workers = False
        counter += nr_batches
        if not data_for_all_workers:
            print("At least one worker ran out of data, stopping.")
            break

        model = utils.federated_avg(models)
        batches = get_next_batches(federated_train_loader, nr_batches)
    return model

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def experiment(num_workers,no_cuda):

    # Creating num_workers clients
    clients = []
    hook = sy.TorchHook(torch)
    clients_mem = torch.zeros(num_workers)
    for i in range(num_workers):
        clients.append(sy.VirtualWorker(hook, id="c "+str(i)))


    # Initializing arguments, with GPU usage or not
    args = Arguments(no_cuda)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
    # TODO Quickhack. Actually need to fix the problem moving the model to CUDA\n",
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}


    # Federated data loader
    federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader
      datasets.MNIST('../data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ]))
      .federate(clients), # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
      batch_size=args.batch_size, shuffle=True,iter_per_worker=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
      datasets.MNIST('../data', train=False, transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ])),
      batch_size=args.test_batch_size, shuffle=True, **kwargs)


    #creating the models for each client
    #models,optimizers = [], []
    #print(device)
    #for i in range(num_workers):
        #print(i)
    #    models.append(Net1().to(device))
    #    models[i] = models[i].send(clients[i])
    #    optimizers.append(optim.SGD(params=models[i].parameters(),lr=0.1))



    start = time.time()
    #%%time
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr) # TODO momentum is not supported at the moment

    for epoch in range(1, args.epochs + 1):
        model = train(args, model, device, federated_train_loader, args.lr, args.federate_after_n_batches, epoch, clients_mem)
        test(args, model, device, test_loader)
        t = time.time()
        print(t-start)
    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    end = time.time()
    print(end - start)
    print("Memory exchanged : ",clients_mem)
    return clients_mem


experiment(num_workers,False)
