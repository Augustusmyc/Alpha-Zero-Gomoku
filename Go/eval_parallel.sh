#!/bin/bash
# eval_parallel.sh  mode  game_num
mode=$1
game_num=${2:-10}

timestamp() { date '+%F %T'; }

tmp_dir=$(mktemp -d /tmp/az_eval_XXXXXX)
trap "rm -rf $tmp_dir" EXIT

# 1. 并行跑 game_num 局
for ((i=0;i<game_num;i++)); do
    ./train_eval_net eval_one_${mode} $((i % 2))
    echo $? > "$tmp_dir"/res_$i.txt &
done
wait

# 2. 汇总
a_win=0 b_win=0 tie=0
for ((i=0;i<game_num;i++)); do
    code=$(cat "$tmp_dir"/res_$i.txt)
    case $code in
        0) ((a_win++)) ;;
        1) ((b_win++)) ;;
        2) ((tie++))   ;;
    esac
done

# 3. 按旧格式写 logger.txt（带时间戳）
cur=$(awk '{print $1}' current_and_best_weight.txt)

if [ "$mode" = "winner" ]; then
    best=$(awk '{print $2}' current_and_best_weight.txt)
    printf "[%s] %d-th weight win: %d  %d-th weight win: %d  tie: %d\n" \
           "$(timestamp)" "$cur" "$a_win" "$best" "$b_win" "$tie" >> logger.txt

    # 更新 best 权重（保持旧逻辑）
    ratio=$(( 1000 * a_win / (b_win + 1) ))   # 千分比，省掉小数
    if (( ratio > 1200 )); then
        echo "$cur $cur" > current_and_best_weight.txt
        printf "[%s] new best weight: %d generated!\n" "$(timestamp)" "$cur" >> logger.txt
    fi
else
    nn_sim=$(awk '{print $2}' mcts_number.txt)
    rnd_sim=$(awk '{print $1}' mcts_number.txt)
    printf "[%s] %d-th weight with mcts [%d] win: %d  Random mcts [%d] win: %d  tie: %d\n" \
           "$(timestamp)" "$cur" "$nn_sim" "$a_win" "$rnd_sim" "$b_win" "$tie" >> logger.txt

    # 更新 mcts_number（保持旧逻辑）
    if [ "$a_win" -eq "$game_num" ]; then
        if [ "$rnd_sim" -lt 8000 ]; then
            new_rnd=$((rnd_sim + 100))
            new_nn=$nn_sim
            printf "[%s] increase random mcts number to: %d\n" "$(timestamp)" "$new_rnd" >> logger.txt
        elif [ "$nn_sim" -gt 17 ]; then
            new_rnd=$rnd_sim
            new_nn=$((nn_sim - 1))
            printf "[%s] decrease nn mcts number to: %d\n" "$(timestamp)" "$new_nn" >> logger.txt
        else
            new_rnd=$rnd_sim
            new_nn=$nn_sim
        fi
        echo "$new_rnd $new_nn" > mcts_number.txt
    fi
fi
