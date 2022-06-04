#include <iostream>
#include <mcts.h>

#include <onnx.h>
#include <play.h>
#include <common.h>

int main() {
  NeuralNetwork *model = new NeuralNetwork(8);
  //torch::optim::SGD optimizer(model->module->parameters(), /*lr=*/0.01);
  SelfPlay *sp = new SelfPlay(model);
  auto train_buffer = sp->self_play_for_train(3);
  std::cout << "3 train size = " << std::get<0>(train_buffer).size() << " " <<
      std::get<1>(train_buffer).size() << " " << std::get<2>(train_buffer).size() << std::endl;
  model->train(std::get<0>(train_buffer), std::get<1>(train_buffer),std::get<2>(train_buffer));
  model->save_weights("1.pt");
  return 0;
}

