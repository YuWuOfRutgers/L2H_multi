import re
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt

dataset_root = Path(sys.argv[1])

result_path = dataset_root.parent / f"{dataset_root.name}.log"
with open(result_path, 'r') as f:
    lines = f.readlines()

START = re.compile(
    r"Training EdgeNet with async_freq=(\d+) on (\w+) with cost_1=([\.\d]+) and cost_e=([\.\d]+)"
)

LOSS = re.compile(
    r"\[(\d+),\s+(\d+)\] loss_edge: ([\.\d]+) loss_rejector: ([\.\d]+)"
)

data = {}

for line in lines:
    if START.search(line):
        async_freq, dataset, cost_1, cost_e = START.search(line).groups()
        async_freq = int(async_freq)
        # Only for better sorting
        if async_freq == 0:
            async_freq = 100000
        if async_freq == 1:
            async_freq = -1
        cost_1 = float(cost_1)
        cost_e = float(cost_e)
        data[(async_freq, dataset, cost_1, cost_e)] = []

    if LOSS.search(line):
        epoch, step, loss_edge, loss_rejector = LOSS.search(line).groups()
        epoch = int(epoch)
        step = int(step)
        loss_edge = float(loss_edge)
        loss_rejector = float(loss_rejector)
        data[(async_freq, dataset, cost_1, cost_e)].append([epoch, step, loss_edge, loss_rejector])

data_plot = OrderedDict()

for key, value in data.items():
    async_freq, dataset, cost_1, cost_e = key
    if (cost_1, cost_e) not in data_plot:
        data_plot[(cost_1, cost_e)] = OrderedDict()

    data_1 = data_plot[(cost_1, cost_e)]
    if dataset not in data_1:
        data_1[dataset] = {}
    if async_freq not in data_1[dataset]:
        data_1[dataset][async_freq] = []

    data_1[dataset][async_freq] = value

data_plot = OrderedDict(sorted(data_plot.items(), key=lambda x: x[0]))
for key, value in data_plot.items():
    data_plot[key] = OrderedDict(sorted(value.items(), key=lambda x: x[0]))
    for dataset, value in data_plot[key].items():
        data_plot[key][dataset] = OrderedDict(sorted(value.items(), key=lambda x: x[0]))

data_1 = data_plot.pop((1.25, 0.25))
plt.figure(figsize=(8, 3))

for i, (dataset, value) in enumerate(data_1.items()):
    plt.subplot(1, 2, i + 1)
    for async_freq, dp in value.items():
        loss = [l[2] + l[3] for l in dp]
        if async_freq == -1:
            plt.plot(loss, label=f"Synchronous", linewidth=.8)
        elif async_freq == 100000:
            plt.plot(loss, label=f"$S=|D|$", linewidth=.8)
        else:
            plt.plot(loss, label=f"$S={async_freq}$", linewidth=.8)
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"{dataset}")

plt.tight_layout()
plt.savefig(dataset_root / "loss.pdf")


plt.figure(figsize=(8, 5))
for i, (cost, value) in enumerate(data_plot.items()):
    for j, (dataset, value) in enumerate(value.items()):
        plt.subplot(2, 2, i * 2 + j + 1)
        for async_freq, dp in value.items():
            loss = [l[2] + l[3] for l in dp]
            if async_freq == -1:
                plt.plot(loss, label=f"Synchronous", linewidth=.8)
            elif async_freq == 100000:
                plt.plot(loss, label=f"$S=|D|$", linewidth=.8)
            else:
                plt.plot(loss, label=f"$S={async_freq}$", linewidth=.8)
        plt.legend()
        plt.yscale("log")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(f"{dataset} ($c_1={cost[0]}, c_e={cost[1]}$)")

plt.tight_layout()
plt.savefig(dataset_root / "loss_all.pdf")
