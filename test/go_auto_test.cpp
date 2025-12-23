#include "go.h"
#include <iostream>
#include <vector> // 必须添加

int main() {
    unsigned int width = 2, height = 2;
    
    Go game(width, height, Go::Black);
    
    std::cout << "围棋程序启动 - 训练模式\n";
    std::cout << "棋盘: " << width << "x" << height << std::endl;
    
    int step = 0;
    while (true) {
        game.render();
        
        auto status = game.get_game_status();
        if (status.first == 1) {
            // 处理平局
            if (status.second == 0) {
                std::cout << "\n终局! 平局!" << std::endl;
            } else {
                std::string winner = (status.second == Go::Black) ? "黑方" : "白方";
                std::cout << "\n终局! " << winner << "获胜!" << std::endl;
            }
            break;
        }
        
        auto moves = game.get_legal_moves();
        
        // 收集所有合法着法的索引
        std::vector<int> legal_moves;
        for (int i = 0; i < (int)moves.size(); i++) {
            if (moves[i] == 1) {
                legal_moves.push_back(i);
            }
        }
        
        if (legal_moves.empty()) {
            std::cout << "\n自动虚手 - 终局检测" << std::endl;
            break;
        }
        
        // 随机选择合法着法
        int move = legal_moves[rand() % legal_moves.size()];
        unsigned int y = move / width;
        unsigned int x = move % width;
        
        try {
            game.execute_move(move);
            std::cout << "步数: " << ++step 
                      << "  落子: " << char('A' + y) << (x + 1) << std::endl;
        } catch (const std::exception& e) {
            std::cout << "非法着法: " << e.what() << "，跳过" << std::endl;
        }
    }
    
    return 0;
}