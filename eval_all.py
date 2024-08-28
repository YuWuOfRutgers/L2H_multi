import argparse
import json
import multiprocessing.pool
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10", help="dataset to use")
parser.add_argument("--save_dir", type=str, default="models", help="directory to save the model")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
args = parser.parse_args()

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)
dataset = args.dataset

all_models = list(save_dir.glob(f"{dataset}-edgenet-*.pth"))

def run_train_with_conf_score(model_path):
    proc = subprocess.run([
        "python",
        "eval_edge.py",
        "--model", str(model_path),
        "--dataset", dataset,
        "--save_dir", args.save_dir
    ], capture_output=True)

    if proc.returncode != 0:
        print(proc.stderr.decode("utf-8"))
        return None

    return json.loads(proc.stdout.decode("utf-8"))

runs = list(all_models)
pool = multiprocessing.pool.ThreadPool(args.num_workers)
result = pool.map(run_train_with_conf_score, runs)
result = {k.stem: v for k, v in zip(runs, result)}

with open(save_dir / f"{dataset}-eval_all.json", "w") as f:
    f.write(json.dumps(result, indent=4))
