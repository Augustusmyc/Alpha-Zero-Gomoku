#include <iostream>
#include <mcts.h>
#include <gomoku.h>
#include <onnx.h>

using namespace std;

int main(int argc, char* argv[]) {
  int board_size = BORAD_SIZE;
  auto g = std::make_shared<Gomoku>(board_size, N_IN_ROW, BLACK);
  
  NeuralNetwork* module = nullptr;
  bool ai_black = true;
  if (argc <= 1) {
      cout << "Warning: Find No weight path and color, assume they are mymodel and 1 (AI color:Black)" << endl;
  #ifdef _WIN32
    module = new NeuralNetwork("E:/Projects/AlphaZero-Onnx/python/mymodel.onnx", NUM_MCT_SIMS);
  #else
    module = new NeuralNetwork("/data/myc/Alpha-Zero-Gomoku/model/423.onnx", NUM_MCT_SIMS);
  #endif
    }
  else {
      ai_black = strcmp(argv[2], "1") == 0 ? true : false;
      string color = ai_black ? "BLACK" : "WHITE";
      cout << "Load weights: "<< argv[1] << "  AI color: " << color << endl;

      module = new NeuralNetwork(argv[1], NUM_MCT_SIMS);
  }
  
  MCTS m(module, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);

  std::cout << "Running..." << std::endl;


  std::pair<int, int> game_state;
  if (ai_black) {
      int res = m.get_best_action(g.get());
      m.update_with_move(res);
      g->execute_move(res);
  }

  while (true) {
    g->render();
    game_state = g->get_game_status();
    if (game_state.first != 0) break;

    // int x, y;
    printf("your move: \n");
    std::string input;
    std::cin >> input;
    
    // 清理输入流状态（防止无限循环）
    if (std::cin.fail()) {
        std::cin.clear(); // 清除错误状态
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // 丢弃无效输入
    }

    // 检查输入长度是否合法
    if (input.length() < 2) {
        std::cout << "输入格式错误，请重新输入!" << std::endl;
        continue;
    }

    // 解析坐标（支持大小写）
    char move_ic = std::toupper(input[0]);
    std::string num_str = input.substr(1);
    
    // 检查字母是否越界
    if (move_ic < 'A' || move_ic >= 'A' + board_size) {
        std::cout << "行号超出范围，有效范围是 A-" << char('A' + board_size - 1) << std::endl;
        continue;
    }
    // 解析数字部分
      try {
          int move_j_int = std::stoi(num_str);
          if (move_j_int < 1 || move_j_int > board_size) {
              std::cout << "列号超出范围，有效范围是 1-" << board_size << std::endl;
              continue;
          }
          
          uint x = move_ic - 'A';
          uint y = move_j_int - 1;
          
          // 检查位置是否非法（修正了拼写错误）
          if (g->is_illegal(x, y)) {
              std::cout << "该位置非法，请重新输入!" << std::endl;
              continue;
          }
          
          // 执行落子
          int my_move = x * board_size + y;
          m.update_with_move(my_move);
          g->execute_move(my_move);
          game_state = g->get_game_status();
          if (game_state.first != 0) {
              g->render();
              break;
          }
          int res = m.get_best_action(g.get());
          m.update_with_move(res);
          g->execute_move(res);
          
      } catch (const std::exception& e) {
          std::cout << "输入格式错误，请重新输入!" << std::endl;
      }
  }
  std::cout << "winner num = " << game_state.second << std::endl;
  return 0;
}

