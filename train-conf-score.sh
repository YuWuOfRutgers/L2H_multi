DATASETS="cifar10 SVHN"

RUN_NAME="$1"

rm -rf "results/$RUN_NAME/code"
mkdir -p "results/$RUN_NAME/code"
cp *.sh "results/$RUN_NAME/code/"
cp *.py "results/$RUN_NAME/code/"
ln -s "../../../data" "results/$RUN_NAME/code/data"
ln -s "../../../results" "results/$RUN_NAME/code/results"

for dataset in $DATASETS; do
    python train_localnet.py --dataset $dataset --save_dir "results/$RUN_NAME/$dataset"
    python train_with_conf_score.py --dataset $dataset --save_dir "results/$RUN_NAME/$dataset" --use_async --async_freq 1
done
