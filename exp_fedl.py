import sys
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

num_workers = int(sys.argv[-1])

# Parameters for the training
class Arguments():
    def __init__(self, no_cuda):
        self.batch_size = 256
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = no_cuda
        self.seed = 1
        self.log_interval = 30
        self.save_model = False


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

# Returns the size of the current model
def model_size(model):
    size = 0
    for p in model.parameters():
        size += p.element_size()*p.nelement()
    return size

def train(args, model, device, federated_train_loader, optimizer, epoch, clients_mem):
    model.train()
    #distributed dataset
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        #send the model to the right location
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        # associate the worker with the right number
        i = int(data.location.id.split()[-1])

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        #send the model back
        model.get()
        # add twice the size of the model: back and forth
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


    # Measuring training time
    start = time.time()

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, federated_train_loader, optimizer, epoch, clients_mem)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    end = time.time()
    print(end - start)
    # Printing the memory exchanged for each client
    print("Memory exchanged : ",clients_mem)
    return clients_mem


experiment(num_workers,False)
