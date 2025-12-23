#include "go.h"
#include <iostream>
#include <cctype>
#include <limits>
#include <string>

int main() {
    unsigned int width = 9, height = 9; // 9x9适合训练
    
    Go game(width, height, Go::Black);
    
    std::cout << "围棋程序启动 - 训练模式\n";
    std::cout << "棋盘: " << width << "x" << height << std::endl;
    
    int step = 0;
    while (true) {
        game.render();
        
        auto status = game.get_game_status();
        if (status.first == 1) {
            std::string winner = (status.second == Go::Black) ? "黑方" : "白方";
            std::cout << "\n终局! " << winner << "获胜!" << std::endl;
            break;
        }
        
        // AlphaZero会在这里调用神经网络选择着法
        // 这里模拟随机落子用于测试
        auto moves = game.get_legal_moves();
        
        if (moves.empty()) {
            std::cout << "\n自动虚手 - 终局检测" << std::endl;
            // 强制结束
            break;
        }
        
        // 随机选择一个着法（实际应由AI决定）
        int move = moves[rand() % moves.size()];
        unsigned int y = move / width;
        unsigned int x = move % width;
        
        try {
            game.execute_move(move);
            std::cout << "步数: " << ++step 
                      << "  落子: " << char('A' + y) << (x + 1) << std::endl;
        } catch (...) {
            std::cout << "非法着法，跳过" << std::endl;
        }
    }
    
    return 0;
}