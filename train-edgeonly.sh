DATASETS="cifar10 SVHN"

RUN_NAME="edgeonly"

rm -rf "results/$RUN_NAME/code"
mkdir -p "results/$RUN_NAME/code"
cp *.sh "results/$RUN_NAME/code/"
cp *.py "results/$RUN_NAME/code/"
ln -s "../../../data" "results/$RUN_NAME/code/data"
ln -s "../../../results" "results/$RUN_NAME/code/results"

for dataset in $DATASETS; do
    python train_edgenet_only.py --dataset $dataset --save_dir "results/$RUN_NAME"
done
