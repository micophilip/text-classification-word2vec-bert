####################################
# Template for running experiments #
####################################

for experiment in {1..10}; do
  python ml/main.py \
    --model_dir /path/to/models \
    --data_dir /path/to/data \
    --train --batch_size=32 \
    --test &>/path/to/logs/experiment-$(date "+%Y.%m.%d-%H.%M.%S").log
done
