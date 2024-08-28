import json
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fetch_data(result_root, dataset_list):
    samples = {}

    for dataset in dataset_list:
        result_path = result_root / dataset
        with open(result_path / f"{dataset}-eval_all.json") as f:
            results = json.load(f)

        samples[dataset] = OrderedDict({
            "$c_1=1$": [],
            "$c_1=1.25$": [],
        })
        for model, result in results.items():
            costs = model.split("-")
            cost_1 = float(costs[3])
            cost_e = float(costs[5])

            reject_rate = result["both"]["predict_remotely"] / result["both"]["total"]
            accuracy = result["both"]["correct"] / result["both"]["total"]

            if cost_1 == 1:
                samples[dataset]["$c_1=1$"].append((reject_rate, accuracy, cost_e))
            if cost_1 == 1.25:
                samples[dataset]["$c_1=1.25$"].append((reject_rate, accuracy, cost_e))

    return samples


if __name__ == "__main__":
    result_root = Path(sys.argv[1])
    dataset_list = ["cifar10", "SVHN"]

    samples = fetch_data(result_root, dataset_list)
    plt.figure(figsize=(12, 5))
    for i, dataset in enumerate(dataset_list):
        for j, (key, value) in enumerate(samples[dataset].items()):
            value = sorted(value, key=lambda x: x[0])
            plt.subplot(2, 4, i * 2 + j + 1)
            x, y, cost = zip(*value)
            fit = np.polyfit(x, y, 3)
            fit_fn = np.poly1d(fit)
            plt.plot(x, fit_fn(x), "--", c="red")
            plt.scatter(x, y, s=2)
            plt.title(f"{dataset} ({key})")
            plt.xlabel("Reject Rate")
            plt.ylabel("Accuracy")

    for i, dataset in enumerate(dataset_list):
        for j, (key, value) in enumerate(samples[dataset].items()):
            value = sorted(value, key=lambda x: x[2])
            plt.subplot(2, 4, i * 2 + j + 1 + 4)
            x, y, cost = zip(*value)
            plt.plot(cost, x)
            plt.title(f"{dataset} ({key})")
            plt.xlabel("$c_e$")
            plt.ylabel("Reject Rate")

    plt.tight_layout()
    plt.savefig(result_root / "reject_rate.pdf")
