import argparse
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import optim

from dataset import load_data
from models import EdgeNetAndRejector, LocalNet
from test_utils import get_accuracy, test_edge

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10", help="dataset to use")
parser.add_argument("--save_dir", type=str, default="models", help="directory to save the model")
parser.add_argument("--cost_1", type=float, default=1.2, help="cost for the rejector")
parser.add_argument("--cost_e", type=float, default=0.2, help="cost for the edge")
parser.add_argument("--async_freq", type=int, default=10, help="frequency of training the rejector")
args = parser.parse_args()

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
dataset = args.dataset

# dataset pre-prossessing
Batch_size=16
Batch_size_test=16
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
print(f"Using {device} device")

torch.manual_seed(43)
np.random.seed(43)
random.seed(43)
# torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(43)

trainset_full, testset_full = load_data(dataset)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset_full, batch_size=Batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset_full, batch_size=Batch_size_test, shuffle=False, num_workers=2)

mobile_net = LocalNet(len(trainset_full.classes)).to(device)
mobile_net.load_state_dict(torch.load(save_dir / f'{dataset}-localnet.pth'))

edge_net = EdgeNetAndRejector(len(trainset_full.classes)).to(device)
edge_net_local = deepcopy(edge_net)
optimizer_edge = optim.SGD(edge_net.edge_net.parameters(), lr=0.001, momentum=0.9)
optimizer_rejector = optim.SGD(edge_net_local.rejector.parameters(), lr=0.0001, momentum=0.9)

# Change this cost
c1 = args.cost_1
ce = args.cost_e

# custom loss function
def surrogate_loss_1(outputs, labels, mobile_outputs):
    # here labels={-1. +1}, so we have to pre-processing before input labels
    r, e = outputs[0], outputs[1]

    loss = torch.nn.functional.cross_entropy(e, labels)
    return loss.sum()

def surrogate_loss_2(outputs, labels, mobile_outputs):
    r, e = outputs[0], outputs[1]
    mobil_predicted = torch.argmax(mobile_outputs, dim=1)
    edge_predicted = torch.argmax(e, dim=1)

    loss = - (labels == mobil_predicted).float() * torch.log(torch.exp(r[:, 0]) / (torch.exp(r[:, 0]) + torch.exp(r[:, 1]))) \
        - (1 - ce + c1 * (labels != edge_predicted).float()) * torch.log(torch.exp(r[:, 1]) / (torch.exp(r[:, 0]) + torch.exp(r[:, 1])))

    return loss.sum()

# training
for epoch in range(15):  # loop over the dataset multiple times [2, 5, 10,30]
    running_loss_edge = 0.0
    running_loss_rejector = 0.0
    num_batches = len(trainloader)
    for i, data in enumerate(trainloader):
        # get the inputs
        # inputs, labels = data
        inputs, labels = data[0].to(device),data[1].to(device)

        with torch.no_grad():
            mobile_outputs = mobile_net(inputs) # this part is ok

        # First train the edge classifier
        optimizer_edge.zero_grad()
        x = edge_net.edge_net(inputs)
        loss_1 = surrogate_loss_1([None, x], labels, None).to(device)
        loss_1.backward()
        optimizer_edge.step()

        if args.async_freq > 0 and i % args.async_freq == 0:  # This controls how often the edge_net is updated to local.
            # Then train the rejector
            edge_net_local.edge_net.load_state_dict(edge_net.edge_net.state_dict())
        if args.async_freq == 0 and i == num_batches - 1:
            edge_net_local.edge_net.load_state_dict(edge_net.edge_net.state_dict())

        optimizer_rejector.zero_grad()
        y = edge_net_local.rejector(inputs)
        with torch.no_grad():
            x = edge_net_local.edge_net(inputs)
        loss_2 = surrogate_loss_2([y, x], labels, mobile_outputs).to(device)
        loss_2.backward()
        optimizer_rejector.step()

        # print statistics
        running_loss_edge += loss_1.item()
        running_loss_rejector += loss_2.item()
        if i % 500 == 499:    # print every 500 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss_edge: {running_loss_edge / 500:.3f} loss_rejector: {running_loss_rejector / 500:.3f}')
            running_loss_edge = 0.0
            running_loss_rejector = 0.0

print('Finished Training')
edge_net.rejector.load_state_dict(edge_net_local.rejector.state_dict())
torch.save(edge_net.state_dict(), save_dir / f'{dataset}-edgenet-async-{args.async_freq}-cost_1-{c1}-cost_e-{ce}.pth')

# testing
labels, predicted_edge_r, predicted_edge_e = test_edge(edge_net, testloader)
total, correct, labels_map = get_accuracy(predicted_edge_e, labels)
print(f"EdgeNet: {correct}/{total} = {correct/total:.2f}")
