import argparse
import random
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

# Load train and test set
trainset_full, testset_full = load_data(dataset)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset_full, batch_size=Batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset_full, batch_size=Batch_size_test, shuffle=False, num_workers=2)

mobile_net = LocalNet(len(trainset_full.classes)).to(device)
mobile_net.load_state_dict(torch.load(save_dir / f'{dataset}-localnet.pth'))

edge_net = EdgeNetAndRejector(len(trainset_full.classes)).to(device)
optimizer = optim.SGD(edge_net.parameters(), lr=0.0001, momentum=0.9)

# Change this cost
c1 = args.cost_1
ce = args.cost_e

# custom loss function
def surrogate_loss(outputs, labels, mobile_outputs):
    # here labels={-1. +1}, so we have to pre-processing before input labels
    r, e = outputs[0], outputs[1]
    mobil_predicted = torch.argmax(mobile_outputs, dim=1)
    edge_predicted = torch.argmax(e, dim=1)

    loss = torch.nn.functional.cross_entropy(e, labels) \
        - (labels == mobil_predicted).float() * torch.log(torch.exp(r[:, 0]) / (torch.exp(r[:, 0]) + torch.exp(r[:, 1]))) \
        - (1 - ce + c1 * (labels != edge_predicted).float()) * torch.log(torch.exp(r[:, 1]) / (torch.exp(r[:, 0]) + torch.exp(r[:, 1])))

    return loss.sum()

# training
for epoch in range(15):  # loop over the dataset multiple times [2, 5, 10,30]
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # get the inputs
        # inputs, labels = data
        inputs, labels = data[0].to(device),data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = edge_net(inputs)

        with torch.no_grad():
            mobile_outputs = mobile_net(inputs) # this part is ok

        loss = surrogate_loss(outputs, labels, mobile_outputs).to(device)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 499:    # print every 500 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

print('Finished Training')

torch.save(edge_net.state_dict(), save_dir / f'{dataset}-edgenet-cost_1-{c1}-cost_e-{ce}.pth')

# testing
labels, predicted_edge_r, predicted_edge_e = test_edge(edge_net, testloader)
total, correct, labels_map = get_accuracy(predicted_edge_e, labels)
print(f"EdgeNet: {correct}/{total} = {correct/total:.2f}")
