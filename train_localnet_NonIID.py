import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from dataset import load_data
from models import LocalNet
from test_utils import get_accuracy, test_local

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10", help="dataset to use")
parser.add_argument("--save_dir", type=str, default="models", help="directory to save the model")
args = parser.parse_args()

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
dataset = args.dataset

# dataset pre-prossessing
Batch_size=16
Batch_size_test=16
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")
print(f"Using {device} device")


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
# torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(42)


# Load train and test set
trainset_full, testset_full = load_data(dataset)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset_full, batch_size=Batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset_full, batch_size=Batch_size_test, shuffle=False, num_workers=2)

# loss function, optimizer, backward
mobile_net = LocalNet(len(trainset_full.classes)).to(device)
criterion = torch.nn.functional.cross_entropy
optimizer = optim.SGD(mobile_net.parameters(), lr=0.001, momentum=0.9)

# training
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = mobile_net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

torch.save(mobile_net.state_dict(), save_dir / f'{dataset}-localnet.pth')

# testing
labels, predicted_local = test_local(mobile_net, testloader)
total, correct, labels_map = get_accuracy(predicted_local, labels)
print(f"LocalNet: {correct}/{total} = {correct/total:.2f}")
torch.save(predicted_local, save_dir / f'{dataset}-predicted_local.pth')
