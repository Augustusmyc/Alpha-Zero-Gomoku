#pragma once
#include <tuple>
#include <vector>
#include <set>
#include <utility>
#include <unordered_set>
#include <random>
#include <common.h>

class Go {
public:
    using move_type = int;
    
    Go(unsigned int width, unsigned int height, int first_color);
    
    // 核心方法
    bool has_legal_moves();
    std::vector<int> get_legal_moves();
    void execute_move(move_type move);
    std::pair<int, int> get_game_status();
    void render();
    bool is_illegal(unsigned int x, unsigned int y);
    
    // 辅助方法
    inline unsigned int get_action_size() const { return width * height; }
    inline board_type get_board() const { return board; }
    inline move_type get_last_move() const { return last_move; }
    inline int get_current_color() const { return cur_color; }
    inline unsigned int get_width() const { return width; }
    inline unsigned int get_height() const { return height; }
    inline int get_black_captures() const { return black_captures; }
    inline int get_white_captures() const { return white_captures; }
    
    // 颜色定义
    enum Color {
        Empty = 0,
        Black = 1,
        White = -1
    };
    
private:
    board_type board;
    const unsigned int width;
    const unsigned int height;
    int cur_color;
    move_type last_move;
    int black_captures;
    int white_captures;
    bool game_ended;
    
    // Zobrist哈希相关
    std::vector<std::vector<uint64_t>> zobrist_black;
    std::vector<std::vector<uint64_t>> zobrist_white;
    uint64_t current_hash;
    std::unordered_set<uint64_t> position_history; // 历史局面
    
    // 领地计算
    std::pair<int, int> count_territory();
    void flood_fill_territory(unsigned int x, unsigned int y, 
                              std::set<std::pair<unsigned int, unsigned int>>& visited,
                              std::set<std::pair<unsigned int, unsigned int>>& region,
                              std::set<int>& adjacent_colors);
    
    struct Group {
        std::set<std::pair<unsigned int, unsigned int>> stones;
        std::set<std::pair<unsigned int, unsigned int>> liberties;
    };
    
    Group find_group(unsigned int x, unsigned int y);
    void remove_group(unsigned int x, unsigned int y);
    bool is_suicide(unsigned int x, unsigned int y, int color);
    bool violates_ko(unsigned int x, unsigned int y, int color);
    int count_liberties(unsigned int x, unsigned int y);
};