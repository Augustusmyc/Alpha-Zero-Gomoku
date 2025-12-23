#pragma once
#include <future>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "chinese_chess.h"

#include <iostream>
#include <onnxruntime_cxx_api.h>

class NeuralNetwork {
 public:
  std::shared_ptr<Ort::Session> shared_session;
  using return_type = std::vector<std::vector<double>>;

  NeuralNetwork(std::string model_path, unsigned int batch_size);
  ~NeuralNetwork();

  std::future<return_type> commit(ChineseChess* game);  // commit task to queue
  // static std::vector<float> transorm_game_to_Tensor(ChineseChess* game);
  // static std::vector<float> transorm_board_to_Tensor(board_type board, int last_move, int cur_player, int no_capture_moves);
  unsigned int batch_size;                             // batch size

 private:
  Ort::Env env;
  Ort::MemoryInfo memory_info;
  Ort::AllocatorWithDefaultOptions allocator;
  using task_type = std::pair<std::vector<float>, std::promise<return_type>>; 
  // pair: input board state(float list), output P and V

  std::vector<const char*> input_node_names;
  std::vector<const char*> output_node_names;
  //std::vector<float> input_tensor_values;
  // std::shared_ptr<torch::jit::script::Module> module;

  void infer();  // infer
  

  std::unique_ptr<std::thread> loop;  // call infer in loop
  bool running;                       // is running

  std::queue<task_type> tasks;  // tasks queue
  std::mutex lock;              // lock for tasks queue
  std::condition_variable cv;   // condition variable for tasks queue
  std::vector<int64_t> input_node_dims;

};