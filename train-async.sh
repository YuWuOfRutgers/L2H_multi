DATASETS="cifar10 SVHN"

RUN_NAME="$1"

rm -rf "results/$RUN_NAME/code"
mkdir -p "results/$RUN_NAME/code"
cp *.sh "results/$RUN_NAME/code/"
cp *.py "results/$RUN_NAME/code/"
ln -s "../../../data" "results/$RUN_NAME/code/data"
ln -s "../../../results" "results/$RUN_NAME/code/results"

for dataset in $DATASETS; do
    echo "Training Local on $dataset"
    python train_localnet.py --dataset $dataset --save_dir "results/$RUN_NAME/" &
done

wait

for S in "0" "1" "100" "1000"; do
    for dataset in $DATASETS; do
        mkdir -p "results/$RUN_NAME/S$S/$dataset"
        cp "results/$RUN_NAME"/*.pth "results/$RUN_NAME/S$S/$dataset"
        (
            echo "Training EdgeNet with async_freq=$S on $dataset with cost_1=1 and cost_e=0.25"
            python train_edgenet_async.py --dataset $dataset --save_dir "results/$RUN_NAME/S$S/$dataset" --async_freq $S --cost_1 1 --cost_e 0.25
            echo "Training EdgeNet with async_freq=$S on $dataset with cost_1=1.25 and cost_e=0"
            python train_edgenet_async.py --dataset $dataset --save_dir "results/$RUN_NAME/S$S/$dataset" --async_freq $S --cost_1 1.25 --cost_e 0
            echo "Training EdgeNet with async_freq=$S on $dataset with cost_1=1.25 and cost_e=0.25"
            python train_edgenet_async.py --dataset $dataset --save_dir "results/$RUN_NAME/S$S/$dataset" --async_freq $S --cost_1 1.25 --cost_e 0.25
        ) | tee "results/$RUN_NAME/S$S/$dataset/$dataset.log" &
    done
    wait
done

cat "results/$RUN_NAME"/S{0,1,100,1000}/*/*.log > "results/$RUN_NAME.log"
