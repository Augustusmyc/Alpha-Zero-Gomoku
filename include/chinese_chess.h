#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <map>
#include <common.h>

class ChineseChess {
public:
    using move_type = std::pair<int, int>; // (from_pos, to_pos)

    ChineseChess(int first_color = RED);

    bool has_legal_moves();
    std::vector<move_type> get_legal_moves();
    void execute_move(move_type move);
    std::pair<int, int> get_game_status();
    void display() const;
    void render();
    bool is_illegal(int from_x, int from_y, int to_x, int to_y);

    inline board_type get_board() const { return this->board; }
    inline move_type get_last_move() const { return this->last_move; }
    inline int get_current_color() const { return this->cur_color; }

    // 棋子类型和颜色定义
    enum PieceType {
        EMPTY = 0,
        GENERAL,   // 将/帅
        ADVISOR,   // 士/仕
        ELEPHANT,  // 象/相
        HORSE,     // 马
        CHARIOT,   // 车
        CANNON,    // 炮
        SOLDIER    // 兵/卒
    };

    enum Color {
        RED = 1,
        GREEN = -1
    };

private:
    board_type board;  // 10x9的棋盘
    const int rows = 10;
    const int cols = 9;
    
    int cur_color;     // 当前玩家颜色
    move_type last_move; // 最后一步棋

    // 初始化棋盘
    void init_board();
    
    // 检查将军状态
    bool is_check(int color);
    
    // 棋子移动规则验证
    bool validate_move(int piece, int from_x, int from_y, int to_x, int to_y);
    
    // 渲染辅助函数
    std::string get_piece_symbol(int piece) const;
};