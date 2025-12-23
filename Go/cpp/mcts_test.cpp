#include <iostream>
#include <mcts.h>
#include <onnx.h>


using namespace std;

int main(int argc, char* argv[]) {
    NeuralNetwork* module = nullptr;
    bool ai_first = true;
    const int height = Go::board_height;
    const int width = Go::board_width;
    auto g = std::make_shared<Go>(width, height, Go::FirstColor);
    // int Fast_NUM_MCT_SIMS = 3; # NUM_MCT_SIMS=1600不方便调试就用这个
    
    if (argc <= 1) {
        cout << "Warning: No weight path specified. AI will play randomly." << endl;
    } else {
        if (argc == 2){
            cout << "No color specified. AI will play as First Color (Black)." << endl;
            ai_first = true;
        }else{
            ai_first = strcmp(argv[2], "1") == 0 ? true : false;
        }        
        string color = ai_first ? "黑色" : "白色";
        cout << "Load weights: " << argv[1] << "  AI color: " << color << endl;
        module = new NeuralNetwork(argv[1], NUM_MCT_SIMS);
    }

    // 初始化MCTS参数
    MCTS m(module, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS,  //NUM_MCT_SIMS*5=9000
           g->get_action_size());

    std::cout << "Go begin..." << std::endl;

    // 处理AI先手
    if (ai_first) {
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
        if (g->get_current_color() == Go::FirstColor) {
            cout << "黑方走棋" << endl;
        } else {
            cout << "白方走棋" << endl;
        }
        
        // 处理玩家输入
        if ((ai_first && g->get_current_color() == Go::SecondColor) || 
            (!ai_first && g->get_current_color() == Go::FirstColor)) {
            string inp;
            cout << "请输入您的着法 (例如: A1): ";
            cin >> inp;
            
            // 转换坐标
            uint in_y = toupper(inp[0]) - 'A';
            uint in_x = inp[1] - '1';
            
            // 验证走法
            while (g->is_illegal(in_x, in_y)) {
                g->debug_print_board();  // 打印看是否真的被污染
                g->print_illegal_reason(in_x, in_y);  // 打印非法原因
                cout << "无效着法! 请重新输入: ";
                cin >> inp;
                in_y = toupper(inp[0]) - 'A';
                in_x = inp[1] - '1';
            }
            
            int res = in_y * g->width + in_x;

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
             << (game_state.second == Go::FirstColor ? "黑方" : "白方")
             << "获胜!" << endl;
    }
    
    return 0;
}