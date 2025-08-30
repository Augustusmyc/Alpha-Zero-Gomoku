#include "chinese_chess.h"
#include <iostream>
#include <cctype> // 用于toupper函数

int main() {
    ChineseChess game(ChineseChess::FirstColor);
    
    while (true) {
        game.render();
        
        auto status = game.get_game_status();
        if (status.first == 1) {
            if (status.second == 0) {
                std::cout << "游戏结束: 和棋!" << std::endl;
            } else {
                std::cout << "游戏结束: " 
                          << (status.second == ChineseChess::FirstColor ? "红方" : "黑方")
                          << "获胜!" << std::endl;
            }
            break;
        }
        
        std::cout << (game.get_current_color() == ChineseChess::FirstColor ? "红方" : "黑方")
                  << "走棋，请输入您的着法 (例如: A1 B2)，或输入'undo'悔棋: ";
        
        std::string input;
        std::cin >> input;
        
        if (input == "undo") {
            if (game.undo_move()) {
                std::cout << "悔棋成功!" << std::endl;
            } else {
                std::cout << "无法悔棋!" << std::endl;
            }
            continue;
        }
        
        std::string from = input;
        std::string to;
        std::cin >> to;
        
        // 检查输入格式是否正确
        if (from.length() != 2 || to.length() != 2 || 
            !isalpha(from[0]) || !isdigit(from[1]) ||
            !isalpha(to[0]) || !isdigit(to[1])) {
            std::cout << "输入格式错误，请重新输入!" << std::endl;
            continue;
        }
        
        int from_x = toupper(from[0]) - 'A';
        int from_y = from[1] - '1';
        int to_x = toupper(to[0]) - 'A';
        int to_y = to[1] - '1';
        
        try {
            // game.execute_move_by_pair({from_x * 9 + from_y, to_x * 9 + to_y});
            int move_code = game.calculate_move_code(from_x, from_y, to_x, to_y);
            // std::cout << "from_x: " << from_x << ", from_y: " << from_y << ", to_x: " << to_x << ", to_y: " << to_y << std::endl;
            int from_pos = from_x * ChineseChess::cols + from_y;
            // std::cout << "from_pos: " << from_pos << " move_code: " << move_code << std::endl;
            game.execute_move_by_squeeze_pair({from_pos, move_code});
        } catch (const std::exception& e) {
            std::cout << "无效着法: " << e.what() << std::endl;
        }
    }
    
    return 0;
}