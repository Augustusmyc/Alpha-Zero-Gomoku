#!/bin/bash
# export CUDA_VISIBLE_DEVICES=3
set -euo pipefail  # 开启严格错误检查（命令失败、未定义变量、管道失败时退出）
trap 'echo "Error: $BASH_COMMAND failed with exit code $?" >&2; exit 1' ERR  # 捕获错误并输出详细信息

n=1
batch_num=5
do_initailize=1
if [ $do_initailize == 1 ]
then
	echo "initailizing........"
	./train_eval_net prepare || exit 1  # 显式检查退出状态
	python ../python/learner.py || exit 1
fi

while [  $n -le 500 ]
do
	echo "--------------$n-th train------------------"
	for ((i=0;i<$batch_num;i++));do
		{
		# sleep 3;echo 1>>haha && echo "done!"
		 ./train_eval_net generate $i
		}&
	done
	wait
	python ../python/learner.py train
	./train_eval_net eval_with_winner 10
	./train_eval_net eval_with_random 10
	let n++
done