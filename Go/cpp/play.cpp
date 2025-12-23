#include <play.h>
#include <mcts.h>
// #include <common.h>
#include <onnx.h>

#include<iostream>
#include<fstream>
#include <chrono>
#include <iomanip>

#define BUFFER_LEN 1000 //if debug，can be set very small such as 3
#define v_buff_type std::vector<int> 
#define p_buff_type std::vector<std::vector<float>>

#define board_buff_type std::vector<board_type>

using namespace std;

SelfPlay::SelfPlay(NeuralNetwork *nn):
        nn(nn),
        thread_pool(new ThreadPool(NUM_TRAIN_THREADS))
        {}


void SelfPlay::play(unsigned int saved_id){
    auto g = std::make_shared<Go>(Go::board_width, Go::board_height,Go::FirstColor);
    
    // 尝试一下新方法，直接将整个游戏状态转化为一个tensor
    
    MCTS *mcts = new MCTS(nn, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS, g->get_action_size());
    std::pair<int,int> game_state;
    game_state = g->get_game_status();
    // std::cout << "play begin !!" << std::endl;
    int step = 0;
    board_buff_type board_buffer(BUFFER_LEN, vector<vector<int>>(g->height, vector<int>(g->width)));
    std::vector<std::vector<float>> board_buffer_new(BUFFER_LEN, vector<float>(g->get_input_size()));
    v_buff_type v_buffer(BUFFER_LEN);
    p_buff_type p_buffer(BUFFER_LEN, vector<float>(g->get_action_size()));// = new p_buff_type();
    vector<int> col_buffer(BUFFER_LEN);
    vector<int> last_move_buffer(BUFFER_LEN);
    vector<float> deadline_ratio_buffer(BUFFER_LEN);

    // diri noise
    static std::gamma_distribution<float> gamma(0.3f, 1.0f);
    static std::default_random_engine rng(std::time(nullptr));

    while (game_state.first == 0) {
        // g->render();
        double temp = step < EXPLORE_STEP ? 1 : 0;
        auto action_probs = mcts->get_action_probs(g.get(), temp);
        //auto action_probs = m->get_action_probs(g.get(), 1);
        //int best_action = m->get_best_action_from_prob(action_probs);

        board_type board = g->get_board();

        if (step >= board_buffer.size()) {
            // cout << "add p_buffer.size() to " << p_buffer.size() << endl;
            board_buffer.emplace_back(vector<vector<int>>(g->height, vector<int>(g->width)));
            p_buffer.emplace_back(vector<float>(Go::action_size));
            v_buffer.push_back(0);  // initial number, update later
            col_buffer.push_back(0);
            last_move_buffer.push_back(0);
            deadline_ratio_buffer.push_back(0.0f);

            board_buffer_new.emplace_back(vector<float>(g->get_input_size()));
        }

        for (int i = 0; i < Go::action_size; i++) {
            p_buffer[step][i] = (float)action_probs[i];
        }
        for (int i = 0; i < g->height; i++) {
            for (int j = 0; j < g->width; j++) {
                board_buffer[step][i][j] = board[i][j];
            }
        }
        // cout << board[9][0] << " =5? " << endl;
        board_buffer_new[step] = g->transorm_game_to_Tensor();
        col_buffer[step] = g->get_current_color();
        last_move_buffer[step] = g->get_last_move();
        deadline_ratio_buffer[step] = float(g->get_step()) / g->time_limit;
        
        std::vector<int> lm = g->get_legal_moves();
        double sum = 0;
        for (int i = 0; i < lm.size(); i++) {
            if (lm[i]) {
                double noi = DIRI * gamma(rng);
                //if (is1){
                //    cout << "noi = " << noi << endl;
                //    is1 = false;
                //}
                action_probs[i] += noi;
            //    if (step < EXPLORE_STEP) {
            //        action_probs[i] += DIRI * gamma(rng);
            //    }
            //    else {
            //        action_probs[i] = (i== best_action) + DIRI * gamma(rng);
            //    }
                sum += action_probs[i];
            }
        }
        for (int i = 0; i < lm.size(); i++) {
            if (lm[i]) {
                action_probs[i] /= sum;
            }
        }


        int res = mcts->get_action_by_sample(action_probs);
        mcts->update_with_move(res);
        g->execute_move(res);
        game_state = g->get_game_status();
        step++;
        // cout << "step = " << step << endl;
    }
    // cout << "Self play: total step num = " << step << " winner = " << game_state.second << endl;
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&t), "[%F %T]");
    cout << ss.str() << " Self play: total step num = " << step
         << " winner = " << game_state.second << endl;

    ofstream writter;
    writter.open("./data/data_" + to_string(saved_id), ios::out | ios::binary);
    writter.write(reinterpret_cast<char*>(&step), sizeof(int));

    for (int i = 0; i < step; i++) {
        for (int j = 0; j < g->height; j++) {
            writter.write(reinterpret_cast<char*>(&board_buffer[i][j][0]), g->width * sizeof(int));
        }
    }
    

    for (int i = 0; i < step; i++) {
        writter.write(reinterpret_cast<char*>(&p_buffer[i][0]), Go::action_size * sizeof(float));
        v_buffer[i] = col_buffer[i] * game_state.second;
    }

    writter.write(reinterpret_cast<char*>(&v_buffer[0]), step * sizeof(int));
    writter.write(reinterpret_cast<char*>(&col_buffer[0]), step * sizeof(int));
    writter.write(reinterpret_cast<char*>(&last_move_buffer[0]), step * sizeof(int));
    writter.write(reinterpret_cast<char*>(&deadline_ratio_buffer[0]), step * sizeof(float));

    // writter.write(reinterpret_cast<char*>(&board_buffer_debug[0]), step * g->get_input_size() * sizeof(float));
    for (int i = 0; i < step; i++) {
        writter.write(reinterpret_cast<char*>(&board_buffer_new[i][0]), g->get_input_size() * sizeof(float));
        // cout << "board_buffer_debug[" << i << "]: " << board_buffer_debug[i][0] << endl;
    }

    writter.close();

      //just validation
      //ifstream inlezen;
      //int new_step;
      //inlezen.open("./data/data_"+str(id), ios::in | ios::binary);
      //inlezen.read(reinterpret_cast<char*>(&new_step), sizeof(int));

      //board_buff_type new_board_buffer(new_step, vector<vector<int>>(BORAD_SIZE, vector<int>(BORAD_SIZE)));
      //p_buff_type new_p_buffer(new_step, vector<float>(BORAD_SIZE * BORAD_SIZE));
      //v_buff_type new_v_buffer(new_step);

      //for (int i = 0; i < step; i++) {
      //    for (int j = 0; j < BORAD_SIZE; j++) {
      //        inlezen.read(reinterpret_cast<char*>(&new_board_buffer[i][j][0]), BORAD_SIZE * sizeof(int));
      //    }
      //}

      //for (int i = 0; i < step; i++) {
      //    inlezen.read(reinterpret_cast<char*>(&new_p_buffer[i][0]), BORAD_SIZE * BORAD_SIZE * sizeof(float));
      //}

      //inlezen.read(reinterpret_cast<char*>(&new_v_buffer[0]), step * sizeof(int));
  }



// void SelfPlay::self_play_for_train(unsigned int game_num,unsigned int start_batch_id){
//     std::vector<std::future<void>> futures;
//     for (unsigned int i = 0; i < game_num; i++) {
//         auto future = thread_pool->commit(std::bind(&SelfPlay::play, this, start_batch_id+i));
//         futures.emplace_back(std::move(future));
//     }
//     this->nn->batch_size = game_num * NUM_MCT_THREADS;
//     for (unsigned int i = 0; i < futures.size(); i++) {
//         futures[i].wait();
       
//         this->nn->batch_size = max((unsigned)1, (game_num - i) * NUM_MCT_THREADS);
//     }
// }

void SelfPlay::self_play_for_train(unsigned int game_num,
                                   unsigned int start_batch_id) {
    // 用硬件并发数而不是 game_num
    ThreadPool tp(std::thread::hardware_concurrency());   // e.g. 32
    std::vector<std::future<void>> futures;

    for (unsigned i = 0; i < game_num; ++i) {
        futures.emplace_back(
            tp.commit(std::bind(&SelfPlay::play, this, start_batch_id + i))
        );
    }
    // 固定 batch_size = 总任务数 * 单局线程数
    nn->batch_size = game_num * NUM_MCT_THREADS;

    for (auto& f : futures) f.wait();
}


