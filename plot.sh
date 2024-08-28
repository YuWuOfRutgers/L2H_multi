RUN_NAME="$1"

python table_accuracy.py "results/$RUN_NAME" async > "results/$RUN_NAME/tables.tex"
python plot_reject_rate.py "results/$RUN_NAME"
