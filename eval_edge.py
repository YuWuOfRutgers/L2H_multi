import sys

save_stdout = sys.stdout
sys.stdout = sys.stderr

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dataset import load_data
from models import EdgeNetAndRejector, LocalNet
from test_utils import get_accuracy, get_accuracy_remote, test_edge

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="model to use")
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

edge_net = EdgeNetAndRejector(len(trainset_full.classes)).to(device)
edge_net.load_state_dict(torch.load(args.model))
edge_net.eval();

result = {}

predicted_local = torch.load(save_dir / f'{dataset}-predicted_local.pth')
# print("------------------ Only run on edge ------------------")
labels, predicted_edge_r, predicted_edge_e = test_edge(edge_net, testloader)
total, correct, labels_map = get_accuracy(predicted_edge_e, labels)
result["edge"] = {"total": total, "correct": correct, "labels_map": labels_map}
# print("------------------ Run on both edge and local ------------------")
total, correct, predict_locally, predict_remotely, labels_map = get_accuracy_remote(predicted_local, predicted_edge_r, predicted_edge_e, labels)
result["both"] = {"total": total, "correct": correct, "predict_locally": predict_locally, "predict_remotely": predict_remotely, "labels_map": labels_map}
# print("------------------ Run on both edge and local (inverted) ------------------")
total, correct, predict_locally, predict_remotely, labels_map = get_accuracy_remote(predicted_local, predicted_edge_r[:, [1, 0]], predicted_edge_e, labels)
result["both_inverted"] = {"total": total, "correct": correct, "predict_locally": predict_locally, "predict_remotely": predict_remotely, "labels_map": labels_map}

print(json.dumps(result, indent=4), file=save_stdout)
