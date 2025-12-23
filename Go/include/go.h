#pragma once
#include <iostream>
#include <tuple>
#include <vector>
#include <set>
#include <utility>
#include <unordered_set>
#include <random>

#define v_buff_type std::vector<int> 
#define p_buff_type std::vector<std::vector<float>>
#define board_type std::vector<std::vector<int>>
#define board_buff_type std::vector<board_type>

class Go {
public:
    std::vector<float> transorm_game_to_Tensor();
    std::vector<float> transorm_board_to_Tensor(board_type board, int last_move, int cur_player, int step);
    using move_type = int;
    const unsigned int width;
    const unsigned int height;
    const unsigned int time_limit = 300;
    Go(unsigned int width, unsigned int height, int first_color);



    // 默认值(神经网络用, 记得和python对齐)
    static const int piece_num = 1; // 只有一种类型的棋子
    static const int last_move_channel = 2*piece_num;
    static const int color_channel = 2*piece_num + 1;
    static const int time_channel = 2*piece_num + 2;

    static const unsigned int board_width = 9; // 和python对齐,下同！
    static const unsigned int board_height = 9;
    static const unsigned int input_size = board_width * board_height * (2*piece_num + 3);
    inline static const unsigned int action_size = board_width * board_height;




    // 核心方法
    bool has_legal_moves();
    std::vector<int> get_legal_moves();
    void execute_move(move_type move);
    std::pair<int, int> get_game_status();
    void render();
    bool is_illegal(unsigned int x, unsigned int y);
    void print_illegal_reason(unsigned int x, unsigned int y);
    
    // 辅助方法
    inline unsigned int get_action_size() const { return width * height; }
    inline unsigned int get_input_size() const { return width * height * (2*piece_num + 3); } // 棋盘大小 *（双方棋子类型*2 + last_move_to_xy + 当前玩家颜色 + 时间戳）
    inline board_type get_board() const { return board; }
    inline move_type get_last_move() const { return last_move; }
    inline int get_current_color() const { return cur_color; }
    inline unsigned int get_width() const { return width; }
    inline unsigned int get_height() const { return height; }
    inline int get_black_captures() const { return black_captures; }
    inline int get_white_captures() const { return white_captures; }
    inline unsigned int get_step() const { return position_history.size(); }
    
    // 颜色定义
    enum Color {
        Empty = 0,
        FirstColor = 1,
        SecondColor = -1
    };

    // 打印棋盘状态用于调试
    void debug_print_board() const {
        std::cerr << "[DEBUG BOARD] cur_color=" << cur_color 
                  << " last_move=" << last_move << std::endl;
        for (unsigned int y = 0; y < height; y++) {
            for (unsigned int x = 0; x < width; x++) {
                std::cerr << board[y][x] << " ";
            }
            std::cerr << std::endl;
        }
    }
 // 仅供调试/观战：打印详细得分与局面信息  
    void print_score_detail() const;



    
private:
    move_type ko_move;
    double komi;
    board_type board;
    
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
    std::pair<int, int> count_territory() const;
    void flood_fill_territory(unsigned int x, unsigned int y, 
                              std::set<std::pair<unsigned int, unsigned int>>& visited,
                              std::set<std::pair<unsigned int, unsigned int>>& region,
                              std::set<int>& adjacent_colors) const;
    
    struct Group {
        std::set<std::pair<unsigned int, unsigned int>> stones;
        std::set<std::pair<unsigned int, unsigned int>> liberties;
    };
    
    Group find_group(unsigned int x, unsigned int y);
    void remove_group(unsigned int x, unsigned int y);
    bool is_suicide(unsigned int x, unsigned int y, int color);
    bool violates_ko(unsigned int x, unsigned int y, int color);
    int count_liberties(unsigned int x, unsigned int y);
    // 内部：把算分逻辑也拆出来，供两处复用
    struct ScoreDetail {
        int black_stones;
        int white_stones;
        int black_territory;
        int white_territory;
        int black_captures;
        int white_captures;
        double komi;
        double black_score;   // 含俘虏+领地+棋子
        double white_score;   // 含俘虏+领地+棋子+贴目
        int winner;           // FirstColor/SecondColor/0(平局)
    };
    ScoreDetail calc_score_detail() const;
};