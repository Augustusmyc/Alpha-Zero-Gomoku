#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
set -euo pipefail  # 开启严格错误检查（命令失败、未定义变量、管道失败时退出）
trap 'echo "Error: $BASH_COMMAND failed with exit code $?" >&2; exit 1' ERR  # 捕获错误并输出详细信息

n=1
batch_num=5
game_num=10    
do_initialize=1

timestamp() { date '+%F %T'; }

if [ "$do_initialize" = 1 ]; then
    echo "[$(timestamp)] initializing..."
    ./train_eval_net prepare || exit 1
    python ../python/learner.py || exit 1
fi

while [ $n -le 500 ]; do
    echo "[$(timestamp)] --------------$n-th train------------------"

    # 1. 并行产生 batch_num 个自弈文件
    for ((i=0;i<batch_num;i++)); do
        ./train_eval_net generate $i &
    done
    wait

    # 2. 训练
    echo "[$(timestamp)] -----learning..."
    python ../python/learner.py train

    # 3. 并行评估
    echo "[$(timestamp)] -----evaluating current and best..."
    ./eval_parallel.sh winner $game_num
    echo "[$(timestamp)] -----evaluating current and random..."
    ./eval_parallel.sh random $game_num

    ((n++))
done