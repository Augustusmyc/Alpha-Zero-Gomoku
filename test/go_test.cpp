#include "go.h"
#include <iostream>
#include <cctype>
#include <limits>
#include <string>

#include <algorithm>  // 必须包含！
using namespace std;

int main() {
    unsigned int width, height;
    
    std::cout << "欢迎使用围棋程序!" << std::endl;
    std::cout << "棋盘宽度(列): ";
    std::cin >> width;
    std::cout << "棋盘高度(行): ";
    std::cin >> height;
    
    width = std::min(26u, std::max(1u, width));
    height = std::min(26u, std::max(1u, height));
    
    Go game(width, height, Go::Black);
    
    std::cout << "\n" << width << "x" << height << "棋盘已创建" << std::endl;
    std::cout << "输入格式: 行字母+列数字 (如 A1) 或 'pass'" << std::endl;

    cout << "=== 测试Go类 ===" << endl;

    // // 测试1：初始状态应该有81个合法着法
    // auto moves = game.get_legal_moves();
    // int legal_count = std::count(moves.begin(), moves.end(), 1);
    // // 3. 打印完整矩阵（每行9个，方便看棋盘）
    // cout << "\n--- moves矩阵 ---" << endl;
    // for (unsigned int y = 0; y < game.get_height(); y++) {
    //     for (unsigned int x = 0; x < game.get_width(); x++) {
    //         int idx = y * game.get_width() + x;
    //         cout << moves[idx] << " ";
    //     }
    //     cout << "  // " << char('A' + y) << "行" << endl;
    // }
    // cout << "-----------------" << endl;
    // cout << "初始合法着法数: " << legal_count << " (应=81)" << endl;
    
    int pass_count = 0;
    
    while (true) {
        game.render();
        
        auto status = game.get_game_status();
        if (status.first == 1) {
            std::string winner = (status.second == Go::Black) ? "黑方" : "白方";
            std::cout << "\n游戏结束! " << winner << "获胜!" << std::endl;
            break;
        }
        
        std::string color_str = (game.get_current_color() == Go::Black) ? "黑方" : "白方";
        std::cout << color_str << "落子: ";
        
        std::string input;
        std::cin >> input;
        
        if (std::cin.fail()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            std::cout << "输入错误!" << std::endl;
            continue;
        }
        
        if (input == "pass") {
            pass_count++;
            if (pass_count >= 2) {
                auto final_status = game.get_game_status();
                std::string winner = (final_status.second == Go::Black) ? "黑方" : "白方";
                std::cout << "\n双方虚手! " << winner << "获胜!" << std::endl;
                break;
            }
            std::cout << color_str << "虚手!" << std::endl;
            game.execute_move(-1);
            continue;
        } else {
            pass_count = 0;
        }
        
        if (input.length() < 2) {
            std::cout << "格式错误! 输入如 'A1' 或 'pass'" << std::endl;
            continue;
        }
        
        char row_char = std::toupper(input[0]);
        std::string col_str = input.substr(1);
        
        if (row_char < 'A' || row_char >= 'A' + height) {
            std::cout << "行号错误! 范围: A-" << char('A' + height - 1) << std::endl;
            continue;
        }
        
        try {
            int col_num = std::stoi(col_str);
            if (col_num < 1 || col_num > (int)width) {
                std::cout << "列号错误! 范围: 1-" << width << std::endl;
                continue;
            }
            
            unsigned int x = col_num - 1;
            unsigned int y = row_char - 'A';
            
            if (game.is_illegal(x, y)) {
                game.print_illegal_reason(x,y);
                std::cout << "非法着法! 不能落子" << std::endl;
                continue;
            }
            
            int move = y * width + x;
            game.execute_move(move);
            
        } catch (...) {
            std::cout << "格式错误! 请重试" << std::endl;
        }
    }
    
    return 0;
}
