#include <iostream>
#include <mcts.h>
#include <onnx.h>
// #include "chinese_chess.h"

using namespace std;

int main(int argc, char* argv[]) {
    auto g = std::make_shared<ChineseChess>();
    NeuralNetwork* module = nullptr;
    bool ai_red = true;
    const int rows = ChineseChess::rows;
    const int cols = ChineseChess::cols;
    
    if (argc <= 1) {
        cout << "Warning: No weight path specified. AI will play randomly." << endl;
    } else {
        if (argc == 2){
            cout << "No color specified. AI will play as First Color (Red)." << endl;
            ai_red = true;
        }else{
            ai_red = strcmp(argv[2], "1") == 0 ? true : false;
        }        
        string color = ai_red ? "红色" : "黑色";
        cout << "Load weights: " << argv[1] << "  AI color: " << color << endl;
        module = new NeuralNetwork(argv[1], NUM_MCT_SIMS);
    }

    // 初始化MCTS参数
    MCTS m(module, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS,  //NUM_MCT_SIMS*5=9000
           ChineseChess::action_size);

    std::cout << "中国象棋游戏开始..." << std::endl;

    // 处理AI先手
    if (ai_red) {
        int res = m.get_best_action(g.get());
        // cout << "AI走棋: " << res << endl;
        m.update_with_move(res);
        g->execute_move(res);
    }

    while (true) {
        g->render();
        auto game_state = g->get_game_status();
        if (game_state.first != 0) break;
        
        // 玩家走棋
        if (g->get_current_color() == ChineseChess::FirstColor) {
            cout << "红方走棋" << endl;
        } else {
            cout << "黑方走棋" << endl;
        }
        
        // 处理玩家输入
        if ((ai_red && g->get_current_color() == ChineseChess::SecondColor) || 
            (!ai_red && g->get_current_color() == ChineseChess::FirstColor)) {
            string from, to;
            cout << "请输入您的着法 (例如: A1 B2): ";
            cin >> from >> to;
            
            // 转换坐标
            int from_x = toupper(from[0]) - 'A';
            int from_y = from[1] - '1';
            int to_x = toupper(to[0]) - 'A';
            int to_y = to[1] - '1';
            
            // 验证走法
            while (g->is_illegal(from_x, from_y, to_x, to_y)) {
                cout << "无效着法! 请重新输入: ";
                cin >> from >> to;
                from_x = toupper(from[0]) - 'A';
                from_y = from[1] - '1';
                to_x = toupper(to[0]) - 'A';
                to_y = to[1] - '1';
            }
            int move_code = g->calculate_move_code(from_x, from_y, to_x, to_y);
            
            // int from_pos = from_x * cols + from_y;
            int res = move_code  + (from_x * cols + from_y) * ChineseChess::mode_num;

            m.update_with_move(res);
            // g->execute_move_by_squeeze_pair({from_pos, move_code});
            g->execute_move(res);
        } 
        // AI走棋
        else {
            int res = m.get_best_action(g.get());
            m.update_with_move(res);
            g->execute_move(res);
        }

        // 检查游戏状态
        game_state = g->get_game_status();
        if (game_state.first != 0) {
            g->render();
            break;
        }
    }

    // 游戏结束处理
    auto game_state = g->get_game_status();
    if (game_state.second == 0) {
        cout << "游戏结束: 和棋 (" << g->time_limit <<"步相互无吃子)!" << endl;
    } else {
        cout << "游戏结束: " 
             << (game_state.second == ChineseChess::FirstColor ? "红方" : "黑方")
             << "获胜!" << endl;
    }
    
    return 0;
}