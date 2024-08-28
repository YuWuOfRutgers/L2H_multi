import argparse
import random
from pathlib import Path

import numpy as np
import torch

from dataset import load_data
from models import EdgeNetAndRejector, LocalNet
from test_utils import get_accuracy, get_accuracy_remote, test_edge, test_local


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="model to use")
parser.add_argument("--model_frame", type=str, default="alexnet", help="model framework to use")
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

torch.manual_seed(44)
np.random.seed(44)
random.seed(44)
# torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(44)

trainset_full, testset_full = load_data(dataset)

# Create data loaders
testloader = torch.utils.data.DataLoader(testset_full, batch_size=Batch_size_test, shuffle=False, num_workers=2)

mobile_net = LocalNet(len(trainset_full.classes)).to(device)
mobile_net.load_state_dict(torch.load(save_dir / f'{dataset}-localnet.pth'))
mobile_net.eval();

print("------------------ Only run on local ------------------")
labels, predicted_local = test_local(mobile_net, testloader)
total, correct, labels_map = get_accuracy(predicted_local, labels)
print("Sample size: ", total)
print('Accuracy of the network on test images: %d %%' % (
    100 * correct / total))
print('Accuracy for each class:')
for i, class_name in enumerate(testset_full.classes):
    print(f'{class_name}: {labels_map[i]["correct"] / labels_map[i]["total"] * 100:.2f}%')


edge_net = EdgeNetAndRejector(len(trainset_full.classes), edge_net=args.model_frame).to(device)
edge_net.load_state_dict(torch.load(args.model))
edge_net.eval();

print("------------------ Only run on edge ------------------")
labels, predicted_edge_r, predicted_edge_e = test_edge(edge_net, testloader)
# only run on edge
total, correct, labels_map = get_accuracy(predicted_edge_e, labels)
print("Sample size: ", total)
print('Accuracy of the network on test images: %d %%' % (
    100 * correct / total))
print('Accuracy for each class:')
for i, class_name in enumerate(testset_full.classes):
    print(f'{class_name}: {labels_map[i]["correct"] / labels_map[i]["total"] * 100:.2f}%')


print("------------------ Run on both edge and local ------------------")
total, correct, predict_locally, predict_remotely, labels_map = get_accuracy_remote(predicted_local, predicted_edge_r, predicted_edge_e, labels)
print('Sample size: %d' %total)
print(f"Decision made locally: {predict_locally}")
print(f"Decision made on edge: {predict_remotely}")

print('Accuracy of the network on test images: %d%%' % (
    100 * correct / total))
print('Accuracy for each class:')
for i, class_name in enumerate(testset_full.classes):
    local_acc = labels_map[i]["local_correct"] / labels_map[i]["local_total"] if labels_map[i]["local_total"] > 0 else float("nan")
    remote_acc = labels_map[i]["remote_correct"] / labels_map[i]["remote_total"] if labels_map[i]["remote_total"] > 0 else float("nan")
    print(
        f'{class_name}:\n\t'
        f'Local Total: {labels_map[i]["local_total"]}, Edge Total: {labels_map[i]["remote_total"]}\n'
        f"\tLocal: {local_acc * 100:.2f}%, "
        f"Edge: {remote_acc * 100:.2f}%, "
        f'Overall: {(labels_map[i]["local_correct"] + labels_map[i]["remote_correct"]) / (labels_map[i]["local_total"] + labels_map[i]["remote_total"]) * 100:.2f}%'
    )


# Experiment 7: The test accuracy of rejected data on mobile classifier and test accuracy of non-rejected data on server classifier.
# We use some hack: send the rejected data to the mobile classifier and non-rejected data to the server classifier.
print("------------------ Run on both edge and local (inverted) ------------------")
total, correct, predict_locally, predict_remotely, labels_map = get_accuracy_remote(predicted_local, predicted_edge_r[:, [1, 0]], predicted_edge_e, labels)
print('Sample size: %d' %total)
print(f"Decision made locally: {predict_locally}")
print(f"Decision made on edge: {predict_remotely}")

print('Accuracy of the network on test images: %d%%' % (
    100 * correct / total))
print('Accuracy for each class:')
for i, class_name in enumerate(testset_full.classes):
    local_acc = labels_map[i]["local_correct"] / labels_map[i]["local_total"] if labels_map[i]["local_total"] > 0 else float("nan")
    remote_acc = labels_map[i]["remote_correct"] / labels_map[i]["remote_total"] if labels_map[i]["remote_total"] > 0 else float("nan")
    print(
        f'{class_name}:\n\t'
        f'Local Total: {labels_map[i]["local_total"]}, Edge Total: {labels_map[i]["remote_total"]}\n'
        f"\tLocal: {local_acc * 100:.2f}%, "
        f"Edge: {remote_acc * 100:.2f}%, "
        f'Overall: {(labels_map[i]["local_correct"] + labels_map[i]["remote_correct"]) / (labels_map[i]["local_total"] + labels_map[i]["remote_total"]) * 100:.2f}%'
    )
