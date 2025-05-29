#include "chinese_chess.h"
#include <iostream>

int main() {
    ChineseChess game(ChineseChess::RED);
    
    while (true) {
        game.render();
        
        auto status = game.get_game_status();
        if (status.first == 1) {
            if (status.second == 0) {
                std::cout << "游戏结束: 和棋!" << std::endl;
            } else {
                std::cout << "游戏结束: " 
                          << (status.second == ChineseChess::RED ? "红方" : "黑方")
                          << "获胜!" << std::endl;
            }
            break;
        }
        
        std::cout << (game.get_current_color() == ChineseChess::RED ? "红方" : "黑方")
                  << "走棋，请输入您的着法 (例如: A1 B2): ";
        
        std::string from, to;
        std::cin >> from >> to;
        
        int from_x = toupper(from[0]) - 'A';
        int from_y = from[1] - '1';
        int to_x = toupper(to[0]) - 'A';
        int to_y = to[1] - '1';
        
        try {
            game.execute_move({from_x * 9 + from_y, to_x * 9 + to_y});
        } catch (const std::exception& e) {
            std::cout << "无效着法: " << e.what() << std::endl;
        }
    }
    
    return 0;
}