#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <map>
// #include <common.h>

#define board_type std::vector<std::vector<int>>

class ChineseChess {
public:
    using move_type = std::pair<int, int>; // (from_pos, to_pos)[last_move] or (from_pos, move_action)

    ChineseChess(int first_color = FirstColor);
    bool undo_move(); // 悔棋

    bool has_legal_moves();
    std::vector<int> get_legal_moves();
    // void execute_move_by_pair(move_type move);
    void execute_move_by_squeeze_pair(move_type move);
    int calculate_move_code(int from_x, int from_y, int to_x, int to_y);
    void execute_move(int move);
    std::pair<int, int> get_game_status();
    void display() const;
    void render();
    bool is_illegal(int from_x, int from_y, int to_x, int to_y);
    int no_capture_moves=0; // 记录未吃子的步数 or 单纯记录时间戳

    inline board_type get_board() const { return this->board; }
    inline move_type get_last_move_pair() const { return this->last_move; } // from_xy, to_xy
    inline int get_last_move() const { return this->last_move.second; } // not from_xy, only to_xy
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
        FirstColor = 1,
        SecondColor = -1
    };

    static const int rows = 10;
    static const int cols = 9;
    static const int mode_num = 9+10+8+8; // 单一位置所有可能的action=35
    static const int piece_num = 7;
    static const int time_limit = 1000; // 100步不吃子/200步未结束，游戏平局
    static const int input_size = rows*cols*(2*piece_num + 3); // 棋盘大小 *（双方棋子类型*2 + last_move_to_xy + 当前玩家颜色 + 已有多长时间不吃子）
    static const int last_move_channel = 2*piece_num;
    static const int color_channel = 2*piece_num + 1;
    static const int time_channel = 2*piece_num + 2;
    inline static const int action_size = rows*cols*mode_num;
    std::vector<float> transorm_game_to_Tensor();
    std::vector<float> transorm_board_to_Tensor(board_type board, int last_move, int cur_player, int no_capture_moves);
private:
    board_type board;  // 10x9的棋盘
    int cur_color;     // 当前玩家颜色
    move_type last_move; // 最后一步棋(from_xy to_xy)

    // 初始化棋盘
    void init_board();
    
    // 检查将军状态
    // bool is_check(int color);

    // 检查将帅是否照面
    bool check_general_face(int from_x, int from_y, int to_x, int to_y);
    
    // 棋子移动规则验证
    bool validate_move(int piece, int from_x, int from_y, int to_x, int to_y);
    bool validate_move(int piece, int from_x, int from_y, int mode_code);
    
    // 渲染辅助函数
    std::string get_piece_symbol(int piece) const;

    std::vector<std::vector<std::vector<int>>> board_history; // 保存棋盘历史状态
    std::vector<int> color_history; // 保存当前玩家历史状态
    
};