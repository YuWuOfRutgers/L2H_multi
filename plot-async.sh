RUN_NAME="$1"

(
    for S in "0" "1" "100" "1000"; do
        python table_accuracy.py "results/$RUN_NAME/S$S" async
    done
) > "results/$RUN_NAME/table.tex"

python table_accuracy_allinone.py "results/$RUN_NAME" async > "results/$RUN_NAME/tables-allinone.tex"

python plot_loss.py "results/$RUN_NAME"
