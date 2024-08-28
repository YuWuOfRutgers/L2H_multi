DATASETS="cifar10 SVHN"

RUN_NAME="$1"

for dataset in $DATASETS; do
    python eval_all.py --dataset $dataset --save_dir "results/$RUN_NAME/$dataset"
done
