DATASETS="cifar10 SVHN"

RUN_NAME="$1"

for S in "0" "1" "100" "1000"; do
    for dataset in $DATASETS; do
        python eval_all.py --dataset $dataset --save_dir "results/$RUN_NAME/S$S/$dataset" &
    done
done

wait
