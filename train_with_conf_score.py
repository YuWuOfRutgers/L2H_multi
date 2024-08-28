import argparse
import multiprocessing.pool
import subprocess

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--cost_1_min", type=float, default=1)
parser.add_argument("--cost_1_max", type=float, default=1.5)
parser.add_argument("--cost_e_min", type=float, default=0)
parser.add_argument("--cost_e_max", type=float, default=0.5)
parser.add_argument("--num_samples", type=int, default=21)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--use_async", action="store_true")
args, remain = parser.parse_known_args()


def run_train_with_conf_score(cost):
    cost_1, cost_e = cost
    subprocess.run([
        "python",
        "train_edgenet.py" if not args.use_async else "train_edgenet_async.py",
        "--cost_1",
        str(cost_1),
        "--cost_e",
        str(cost_e),
        *remain
    ])


runs = []
for cost_e in np.round(np.linspace(args.cost_e_min, args.cost_e_max, args.num_samples), 6):
    runs.append((1, cost_e))
    runs.append(((args.cost_1_min + args.cost_1_max) / 2, cost_e))

pool = multiprocessing.pool.ThreadPool(args.num_workers)
pool.map(run_train_with_conf_score, runs)
