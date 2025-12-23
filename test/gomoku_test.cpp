#include <gomoku.h>
#include <iostream>
#include <cctype> // 用于toupper函数
#include <limits> // 用于numeric_limits
#include <string> // 用于字符串处理

int main() {
    // ChineseChess game(ChineseChess::FirstColor);
    int board_size = 15;
    Gomoku game(board_size, 5, Gomoku::FirstColor);
    
    while (true) {
        game.render();
        
        auto status = game.get_game_status();
        if (status.first == 1) {
            if (status.second == 0) {
                std::cout << "游戏结束: 和棋!" << std::endl;
            } else {
                std::cout << "游戏结束: " 
                          << (status.second == Gomoku::FirstColor ? "黑方" : "白方")
                          << "获胜!" << std::endl;
            }
            break;
        }
        
        std::cout << (game.get_current_color() == Gomoku::FirstColor ? "黑方" : "白方")
                  << "走棋，请输入落子处 (例如: A1): ";
        
        std::string input;
        std::cin >> input;
        
        // 清理输入流状态（防止无限循环）
        if (std::cin.fail()) {
            std::cin.clear(); // 清除错误状态
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // 丢弃无效输入
        }
        
        // if (input == "undo") {
        //     if (game.undo_move()) {
        //         std::cout << "悔棋成功!" << std::endl;
        //     } else {
        //         std::cout << "无法悔棋!" << std::endl;
        //     }
        //     continue;
        // }
        
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
            if (game.is_illegal(x, y)) {
                std::cout << "该位置非法，请重新输入!" << std::endl;
                continue;
            }
            
            // 执行落子
            int my_move = x * board_size + y;
            game.execute_move(my_move);
            
        } catch (const std::exception& e) {
            std::cout << "输入格式错误，请重新输入!" << std::endl;
        }
    }
    
    return 0;
}