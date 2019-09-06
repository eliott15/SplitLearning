import sys
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

num_workers = int(sys.argv[-1])
# Parameters of the training
class Arguments():
    def __init__(self, no_cuda):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = no_cuda
        self.seed = 1
        self.log_interval = 30
        self.save_model = False

# SplitNN classes
# First part of the model, will be owned by each worker
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.max_pool2d(x, 2, 2)

# Second part of the model, owned by the server only
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, federated_train_loader, optimizer, epoch, models, optimizers, clients_mem):
    model.train()
    for batch_idx, (data, targs) in enumerate(federated_train_loader):
        # Find the location of data, and select model and optimizer accordingly
        i = int(data.location.id.split()[-1])
        mod_c,opt_c = models[i], optimizers[i]



        # erase previous gradients (if they exist) and update parameters
        # works only in this way
        optimizer.step()
        opt_c.step()
        opt_c.zero_grad()
        optimizer.zero_grad()

        # making copies before sending to avoid PySyft errors
        tg_copy = targs.copy()
        target = tg_copy.get()
        data, target = data.to(device), target.to(device)

        # make a prediction until cut layer (client location)
        pred_c = mod_c(data)
        copy = pred_c.copy()


        # get this to the server: will be the input of the second part of the model
        inp = copy.get()

        # make prediction with second part of the model (server location)
        pred = model(inp)

        # 5) calculate how much we missed
        loss = F.nll_loss(pred, target)
        loss.backward()

        gradient = inp.grad

        # add the size of the output of the cut layer + the size of the gradient
        clients_mem[i] += (gradient.element_size()*gradient.nelement()) + (inp.element_size() * inp.nelement())
        print((gradient.element_size()*gradient.nelement()) + (inp.element_size() * inp.nelement()))

        # Send the gradient to the client for backpropagation
        gradient = gradient.send(data.location)
        pred_c.backward(gradient)


        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
                100. * batch_idx / len(federated_train_loader), loss.item()))


def test(args, model, device, test_loader, models):
    model.eval()
    test_loss = 0
    correct = 0
    n = len(models)
    M = []
    for i in range(n):
        M.append(models[0].copy().get())
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = 0
            for i in range(n):
                output += model(M[i](data))
            output = output/n
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
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
    federated_train_loader = sy.FederatedDataLoader(
      datasets.MNIST('../data', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ]))
      .federate(clients),
      batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
      datasets.MNIST('../data', train=False, transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ])),
      batch_size=args.test_batch_size, shuffle=True, **kwargs)


    #creating the models and optimizers for each client
    models,optimizers = [], []
    for i in range(num_workers):
        models.append(Net1().to(device))
        models[i] = models[i].send(clients[i])
        optimizers.append(optim.SGD(params=models[i].parameters(),lr=0.1))


    # measuring training time
    start = time.time()
    model = Net2().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, federated_train_loader, optimizer, epoch, models, optimizers,clients_mem)
        test(args, model, device, test_loader, models)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    end = time.time()
    print(end - start)
    # printing the memory exchanged by each client 
    print("Memory exchanged : ",clients_mem)
    return clients_mem


experiment(num_workers,False)
