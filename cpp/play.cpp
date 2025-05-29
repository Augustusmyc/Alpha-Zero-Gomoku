#include <play.h>
#include <mcts.h>
#include <gomoku.h>
#include <common.h>
#include <onnx.h>

#include<iostream>
#include<fstream>

using namespace std;

SelfPlay::SelfPlay(NeuralNetwork *nn):
        //p_buffer(new p_buff_type()),
        //board_buffer(new board_buff_type()),
        //v_buffer(new v_buff_type()),
        nn(nn),
        thread_pool(new ThreadPool(NUM_TRAIN_THREADS))
        {}


void SelfPlay::play(unsigned int saved_id){
  auto g = std::make_shared<Gomoku>(BORAD_SIZE, N_IN_ROW, BLACK);
  MCTS *mcts = new MCTS(nn, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
  std::pair<int,int> game_state;
  game_state = g->get_game_status();
  //std::cout << "begin !!" << std::endl;
  int step = 0;
  board_buff_type board_buffer(BUFFER_LEN, vector<vector<int>>(BORAD_SIZE, vector<int>(BORAD_SIZE)));
  v_buff_type v_buffer(BUFFER_LEN);
  p_buff_type p_buffer(BUFFER_LEN, vector<float>(BORAD_SIZE * BORAD_SIZE));// = new p_buff_type();
  vector<int> col_buffer(BUFFER_LEN);
  vector<int> last_move_buffer(BUFFER_LEN);
  // diri noise
  static std::gamma_distribution<float> gamma(0.3f, 1.0f);
  static std::default_random_engine rng(std::time(nullptr));

  while (game_state.first == 0) {
        //g->render();
        double temp = step < EXPLORE_STEP ? 1 : 0;
        auto action_probs = mcts->get_action_probs(g.get(), temp);
        //auto action_probs = m->get_action_probs(g.get(), 1);
        //int best_action = m->get_best_action_from_prob(action_probs);

        board_type board = g->get_board();
        for (int i = 0; i < BORAD_SIZE * BORAD_SIZE; i++) {
            p_buffer[step][i] = (float)action_probs[i];
        }
        for (int i = 0; i < BORAD_SIZE; i++) {
            for (int j = 0; j < BORAD_SIZE; j++) {
                board_buffer[step][i][j] = board[i][j];
            }
        }
        col_buffer[step] = g->get_current_color();
        last_move_buffer[step] = g->get_last_move();

        
        
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
    }
      cout << "Self play: total step num = " << step << " winner = " << game_state.second << endl;

      ofstream bestand;
      bestand.open("./data/data_" + to_string(saved_id), ios::out | ios::binary);
      bestand.write(reinterpret_cast<char*>(&step), sizeof(int));

      for (int i = 0; i < step; i++) {
          for (int j = 0; j < BORAD_SIZE; j++) {
              bestand.write(reinterpret_cast<char*>(&board_buffer[i][j][0]), BORAD_SIZE * sizeof(int));
          }
      }

      for (int i = 0; i < step; i++) {
          bestand.write(reinterpret_cast<char*>(&p_buffer[i][0]), BORAD_SIZE * BORAD_SIZE * sizeof(float));
          v_buffer[i] = col_buffer[i] * game_state.second;
      }

      bestand.write(reinterpret_cast<char*>(&v_buffer[0]), step * sizeof(int));
      bestand.write(reinterpret_cast<char*>(&col_buffer[0]), step * sizeof(int));
      bestand.write(reinterpret_cast<char*>(&last_move_buffer[0]), step * sizeof(int));

      bestand.close();

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



void SelfPlay::self_play_for_train(unsigned int game_num,unsigned int start_batch_id){
    std::vector<std::future<void>> futures;
    for (unsigned int i = 0; i < game_num; i++) {
        auto future = thread_pool->commit(std::bind(&SelfPlay::play, this, start_batch_id+i));
        futures.emplace_back(std::move(future));
    }
    this->nn->batch_size = game_num * NUM_MCT_THREADS;
    for (unsigned int i = 0; i < futures.size(); i++) {
        futures[i].wait();
       
        this->nn->batch_size = max((unsigned)1, (game_num - i) * NUM_MCT_THREADS);
       // cout << "end" << endl;
    }
    //return { *this->board_buffer , *this->p_buffer ,*this->v_buffer };
}

// pair<int,int> SelfPlay::self_play_for_eval(unsigned int game_num, NeuralNetwork* a, NeuralNetwork* b) {
//    std::vector<std::future<void>> futures;
//    //NeuralNetwork* a = new NeuralNetwork(NUM_MCT_THREADS * NUM_MCT_SIMS);
//    for (unsigned int i = 0; i < game_num; i++) {
//        auto future = thread_pool->commit(std::bind(&SelfPlay::play, this,a,b));
//        futures.emplace_back(std::move(future));
//    }
//    this->nn->batch_size = game_num * NUM_MCT_THREADS;
//    for (unsigned int i = 0; i < futures.size(); i++) {
//        futures[i].wait();
//
//        this->nn->batch_size = max((unsigned)1, (game_num - i) * NUM_MCT_THREADS);
//    }
//    //return { *this->board_buffer , *this->p_buffer ,*this->v_buffer };
//}
