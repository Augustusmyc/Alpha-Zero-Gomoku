#!/bin/bash
n=1
batch_num=1
do_prepare=1
if [ $do_prepare == 1 ]
then
	echo "preparing........"
	bash ./train_net.sh prepare
	/data/miniconda3/bin/python ../python/learner.py
fi

while [  $n -le 2000 ]
do
	echo "--------------$n-th train------------------"
	for ((i=0;i<$batch_num;i++));do
		{
		# sleep 3;echo 1>>aa && echo "done!"
		bash ./train_net.sh generate $i
		}&
	done
	wait
	/data/miniconda3/bin/python ../python/learner.py train
#	bash ./train_net.sh eval_with_winner 10
	bash ./train_net.sh eval_with_random 10
	let n++
done