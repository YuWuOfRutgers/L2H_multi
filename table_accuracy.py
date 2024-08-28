import json
import sys
from pathlib import Path

result_root = Path(sys.argv[1])

use_async = False
if len(sys.argv) > 2 and sys.argv[2] == "async":
    use_async = True

dataset_list = ["cifar10", "SVHN"]

accuracy_results = {}

for dataset in dataset_list:
    result_path = result_root / dataset

    with open(result_path / f"{dataset}-eval_all.json") as f:
        results = json.load(f)
    for model, result in results.items():
        costs = model.split("-")
        if not use_async:
            cost_1 = float(costs[3])
            cost_e = float(costs[5])
            key = (cost_1, cost_e)
        else:
            async_freq = int(costs[3])
            cost_1 = float(costs[5])
            cost_e = float(costs[7])
            key = (async_freq, cost_1, cost_e)

        local_total = result["both"]["predict_locally"]
        remote_total = result["both"]["predict_remotely"]
        local_correct = sum([x["local_correct"] for x in result["both"]["labels_map"].values()])
        remote_correct = sum([x["remote_correct"] for x in result["both"]["labels_map"].values()])
        local_inv_correct = sum([x["local_correct"] for x in result["both_inverted"]["labels_map"].values()])
        remote_inv_correct = sum([x["remote_correct"] for x in result["both_inverted"]["labels_map"].values()])

        local_acc = local_correct / local_total if local_total > 0 else "N/A"
        remote_acc = remote_correct / remote_total if remote_total > 0 else "N/A"

        local_deferred_acc = local_inv_correct / remote_total if remote_total > 0 else "N/A"
        remote_deferred_acc = remote_inv_correct / local_total if local_total > 0 else "N/A"

        if key not in accuracy_results:
            accuracy_results[key] = {}

        accuracy_results[key][dataset] = (local_acc, remote_acc, local_deferred_acc, remote_deferred_acc)


for key, dataset_results in accuracy_results.items():
    dataset_list = list(dataset_results.keys())
    print("\\begin{table}")
    print("\\centering""")
    if not use_async:
        cost_1, cost_e = key
        print(f"\\caption{{Experiment Results for $c_1 = {cost_1}$ and $c_e = {cost_e}$}}", end="")
        print("\\label{tab:exp-accuracies}")
    else:
        async_freq, cost_1, cost_e = key
        if async_freq == 1:
            print(f"\\caption{{Training Synchronously with $c_1 = {cost_1}$ and $c_e = {cost_e}$}}", end="")
        elif async_freq == 0:
            print(f"\\caption{{Training Asynchronously with $S = |D|$, $c_1 = {cost_1}$ and $c_e = {cost_e}$}}", end="")
        else:
            print(f"\\caption{{Training Asynchronously with $S = {async_freq}$, $c_1 = {cost_1}$ and $c_e = {cost_e}$}}", end="")
        print("\\label{tab:exp-accuracies-async}")
    print(f"\\begin{{tabular}}{{{'c' * (len(dataset_list) * 3 + 1)}}}")
    print(f"\\toprule")
    print("\\multirow{2}{*}{} & ", end="")
    dataset_str = []
    for dataset in dataset_list:
        dataset_str.append(f"\\multicolumn{{3}}{{c}}{{{dataset} (\\%)}}")
    print(" & ".join(dataset_str) + " \\\\")
    print(f"\\cmidrule{{2-{len(dataset_list) * 3 + 1}}}")
    print(f"& {' & '.join(['m & e & differ.'] * len(dataset_list))} \\\\")
    print("\\midrule")
    print("data with rejector $r(x)=\\textsc{local}$ & ", end="")
    nums_str = []
    for dataset in dataset_list:
        local_acc, remote_acc, local_deferred_acc, remote_deferred_acc = dataset_results[dataset]
        nums_str.append("{} & {} & {}".format(
            f"{local_acc * 100:.1f}" if local_acc != "N/A" else "N/A",
            f"{remote_deferred_acc * 100:.1f}" if remote_deferred_acc != "N/A" else "N/A",
            f"\\textbf{{{(remote_deferred_acc - local_acc) * 100:.1f}}}" if local_acc != "N/A" and remote_deferred_acc != "N/A" else "N/A"
        ))
    print(" & ".join(nums_str) + " \\\\")
    print("data with rejector $r(x)=\\textsc{remote}$ & ", end="")
    nums_str = []
    for dataset in dataset_list:
        local_acc, remote_acc, local_deferred_acc, remote_deferred_acc = dataset_results[dataset]
        nums_str.append("{} & {} & {}".format(
            f"{local_deferred_acc * 100:.1f}" if local_deferred_acc != "N/A" else "N/A",
            f"{remote_acc * 100:.1f}" if remote_acc != "N/A" else "N/A",
            f"\\textbf{{{(remote_acc - local_deferred_acc) * 100:.1f}}}" if local_deferred_acc != "N/A" and remote_acc != "N/A" else "N/A"
        ))
    print(" & ".join(nums_str) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("")
    print("")
