#pragma once
#include <future>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <onnx.h>
#include <thread_pool.h>
#include "chinese_chess.h"
// #include <common.h>

class SelfPlay{
    public:
        SelfPlay(NeuralNetwork *nn);
        //~SelfPlay();
        void play(unsigned int saved_id);

        void self_play_for_train(unsigned int game_num, unsigned int start_batch_id);
        
    private:
        //p_buff_type *p_buffer;
        //board_buff_type *board_buffer;
        //v_buff_type *v_buffer;
        
        NeuralNetwork *nn;
        std::unique_ptr<ThreadPool> thread_pool;
        //std::queue<task_type> tasks;  // tasks queue
        std::mutex lock;              // lock for tasks queue
        std::condition_variable cv;   // condition variable for tasks queue
};
