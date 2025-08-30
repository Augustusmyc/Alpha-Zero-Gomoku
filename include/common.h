#pragma once
#include<iostream>
//#define SMALL_BOARD_MODE
//#define USE_GPU

#define CHANNEL_SIZE 3

#ifdef SMALL_BOARD_MODE
    #define BORAD_SIZE 3
    #define N_IN_ROW 3
    #define NUM_MCT_THREADS 4
    #define NUM_MCT_SIMS 54
    #define EXPLORE_STEP 3
    #define C_PUCT 3
    #define C_VIRTUAL_LOSS 1
    #define NUM_CHANNELS 64
    #define NUM_LAYERS 2

    #define BATCH_SIZE 64 //512
    #define DIRI 0.1

    #define NUM_TRAIN_THREADS 70
#else
    #define BORAD_SIZE 15
    #define N_IN_ROW 5
    #define NUM_MCT_THREADS 4
    #define NUM_MCT_SIMS 1600 // debug mode: 16; std mode:1600
    #define EXPLORE_STEP 10
    #define C_PUCT 5
    #define C_VIRTUAL_LOSS 3

    #define BATCH_SIZE 256
    #define DIRI 0.01

    #define NUM_TRAIN_THREADS 10
#endif

#define BLACK 1
#define WHITE -BLACK




#define BUFFER_LEN BORAD_SIZE*BORAD_SIZE+1
#define v_buff_type std::vector<int> 
#define p_buff_type std::vector<std::vector<float>>
#define board_type std::vector<std::vector<int>>
#define board_buff_type std::vector<board_type>
