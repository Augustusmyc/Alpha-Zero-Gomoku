#include "chinese_chess.h"
#include <iostream>
#include <sstream>
#include <algorithm>

ChineseChess::ChineseChess(int first_color) : cur_color(first_color), last_move({-1, -1}) {
    init_board();
}

void ChineseChess::init_board() {
    board = std::vector<std::vector<int>>(rows, std::vector<int>(cols, EMPTY));
    
    // 红方布局 (下方)
    board[9][0] = board[9][8] = static_cast<int>(CHARIOT) * static_cast<int>(RED);
    board[9][1] = board[9][7] = static_cast<int>(HORSE) * static_cast<int>(RED);
    board[9][2] = board[9][6] = static_cast<int>(ELEPHANT) * static_cast<int>(RED);
    board[9][3] = board[9][5] = static_cast<int>(ADVISOR) * static_cast<int>(RED);
    board[9][4] = static_cast<int>(GENERAL) * static_cast<int>(RED);
    board[7][1] = board[7][7] = static_cast<int>(CANNON) * static_cast<int>(RED);
    board[6][0] = board[6][2] = board[6][4] = board[6][6] = board[6][8] = static_cast<int>(SOLDIER) * static_cast<int>(RED);
    
    // 黑方布局 (上方)
    board[0][0] = board[0][8] = static_cast<int>(CHARIOT) * static_cast<int>(GREEN);
    board[0][1] = board[0][7] = static_cast<int>(HORSE) * static_cast<int>(GREEN);
    board[0][2] = board[0][6] = static_cast<int>(ELEPHANT) * static_cast<int>(GREEN);
    board[0][3] = board[0][5] = static_cast<int>(ADVISOR) * static_cast<int>(GREEN);
    board[0][4] = static_cast<int>(GENERAL) * static_cast<int>(GREEN);
    board[2][1] = board[2][7] = static_cast<int>(CANNON) * static_cast<int>(GREEN);
    board[3][0] = board[3][2] = board[3][4] = board[3][6] = board[3][8] = static_cast<int>(SOLDIER) * static_cast<int>(GREEN);
}

bool ChineseChess::validate_move(int piece, int from_x, int from_y, int to_x, int to_y) {
    int piece_type = abs(piece);
    int color = piece > 0 ? RED : GREEN;
    
    // 基本检查
    if (to_x < 0 || to_x >= rows || to_y < 0 || to_y >= cols) return false;
    if (board[to_x][to_y] != EMPTY && (board[to_x][to_y] * color > 0)) return false;
    
    int dx = to_x - from_x;
    int dy = to_y - from_y;
    
    switch (piece_type) {
        case GENERAL: {
            // 只能在九宫内移动
            if (color == RED && (to_x < 7 || to_x > 9 || to_y < 3 || to_y > 5)) return false;
            if (color == GREEN && (to_x < 0 || to_x > 2 || to_y < 3 || to_y > 5)) return false;
            // 只能走一格直线
            if ((abs(dx) + abs(dy) != 1) || (dx != 0 && dy != 0)) return false;
            
            // 将帅不能直接照面
            if (dy == 0) {
                int general_x = -1, general_y = to_y;
                for (int i = 0; i < rows; ++i) {
                    if (abs(board[i][general_y]) == GENERAL && 
                        board[i][general_y] * color < 0) {
                        general_x = i;
                        break;
                    }
                }
                
                if (general_x != -1) {
                    int start = std::min(to_x, general_x) + 1;
                    int end = std::max(to_x, general_x);
                    bool has_block = false;
                    for (int x = start; x < end; ++x) {
                        if (board[x][general_y] != EMPTY) {
                            has_block = true;
                            break;
                        }
                    }
                    if (!has_block) return false;
                }
            }
            return true;
        }
            
        case ADVISOR: {
            // 只能在九宫内移动
            if (color == RED && (to_x < 7 || to_x > 9 || to_y < 3 || to_y > 5)) return false;
            if (color == GREEN && (to_x < 0 || to_x > 2 || to_y < 3 || to_y > 5)) return false;
            // 只能走一格斜线
            return abs(dx) == 1 && abs(dy) == 1;
        }
            
        case ELEPHANT: {
            // 不能过河
            if (color == RED && to_x < 5) return false;
            if (color == GREEN && to_x > 4) return false;
            // 走田字且不被蹩脚
            if (abs(dx) != 2 || abs(dy) != 2) return false;
            return board[from_x + dx/2][from_y + dy/2] == EMPTY;
        }
            
        case HORSE: {
            // 走日字且不被蹩脚
            if (!((abs(dx) == 2 && abs(dy) == 1) || (abs(dx) == 1 && abs(dy) == 2))) return false;
            if (abs(dx) == 2) return board[from_x + dx/2][from_y] == EMPTY;
            else return board[from_x][from_y + dy/2] == EMPTY;
        }
            
        case CHARIOT: {
            if (dx != 0 && dy != 0) return false;
            // 检查路径上是否有阻挡
            if (dx == 0) { // 横向移动
                int step = dy > 0 ? 1 : -1;
                for (int y = from_y + step; y != to_y; y += step) {
                    if (board[from_x][y] != EMPTY) return false;
                }
            } else { // 纵向移动
                int step = dx > 0 ? 1 : -1;
                for (int x = from_x + step; x != to_x; x += step) {
                    if (board[x][from_y] != EMPTY) return false;
                }
            }
            return true;
        }
            
        case CANNON: {
            if (dx != 0 && dy != 0) return false;
            // 检查路径上是否有阻挡
            int count = 0;
            if (dx == 0) { // 横向移动
                int step = dy > 0 ? 1 : -1;
                for (int y = from_y + step; y != to_y; y += step) {
                    if (board[from_x][y] != EMPTY) count++;
                }
            } else { // 纵向移动
                int step = dx > 0 ? 1 : -1;
                for (int x = from_x + step; x != to_x; x += step) {
                    if (board[x][from_y] != EMPTY) count++;
                }
            }
            // 吃子时需要有一个炮架，移动时不能有阻挡
            if (board[to_x][to_y] == EMPTY) return count == 0;
            else return count == 1;
        }
            
        case SOLDIER: {
            if (color == RED) {
                if (from_x < to_x) return false; // 不能后退
                if (from_x > 4) { // 未过河(初始在第6行)
                    return (from_x - to_x == 1) && (dy == 0); // 只能向前一步
                } else { // 已过河
                    return ((from_x - to_x == 1) && (dy == 0)) || 
                        ((from_x == to_x) && abs(dy) == 1); // 可前或横
                }
            } else { // GREEN
                if (from_x > to_x) return false; // 不能后退
                if (from_x < 5) { // 未过河(初始在第3行)
                    return (to_x - from_x == 1) && (dy == 0); // 只能向前一步
                } else { // 已过河
                    return ((to_x - from_x == 1) && (dy == 0)) || 
                        ((from_x == to_x) && abs(dy) == 1); // 可前或横
                }
            }
        }
            
        default:
            return false;
    }
}

bool ChineseChess::is_check(int color) {
    // 找到将/帅的位置
    int general_x = -1, general_y = -1;
    int general = GENERAL * color;
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (board[i][j] == general) {
                general_x = i;
                general_y = j;
                break;
            }
        }
        if (general_x != -1) break;
    }
    
    if (general_x == -1) return true; // 将/帅被吃掉了
    
    // 检查是否被将军
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (board[i][j] * color < 0) { // 对方棋子
                if (validate_move(board[i][j], i, j, general_x, general_y)) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

bool ChineseChess::is_illegal(int from_x, int from_y, int to_x, int to_y) {
    if (from_x < 0 || from_x >= rows || from_y < 0 || from_y >= cols) return true;
    if (to_x < 0 || to_x >= rows || to_y < 0 || to_y >= cols) return true;
    
    int piece = board[from_x][from_y];
    if (piece == EMPTY) return true;
    if ((piece > 0 && cur_color == GREEN) || (piece < 0 && cur_color == RED)) return true;
    
    return !validate_move(piece, from_x, from_y, to_x, to_y);
}

std::vector<ChineseChess::move_type> ChineseChess::get_legal_moves() {
    std::vector<move_type> legal_moves;
    
    for (int from_x = 0; from_x < rows; ++from_x) {
        for (int from_y = 0; from_y < cols; ++from_y) {
            int piece = board[from_x][from_y];
            if (piece == EMPTY || (piece > 0 && cur_color == GREEN) || (piece < 0 && cur_color == RED)) {
                continue;
            }
            
            for (int to_x = 0; to_x < rows; ++to_x) {
                for (int to_y = 0; to_y < cols; ++to_y) {
                    if (validate_move(piece, from_x, from_y, to_x, to_y)) {
                        // 模拟移动检查是否会导致自己被将军
                        int captured = board[to_x][to_y];
                        board[to_x][to_y] = piece;
                        board[from_x][from_y] = EMPTY;
                        
                        bool in_check = is_check(cur_color);
                        
                        // 恢复棋盘
                        board[from_x][from_y] = piece;
                        board[to_x][to_y] = captured;
                        
                        if (!in_check) {
                            legal_moves.emplace_back(from_x * cols + from_y, to_x * cols + to_y);
                        }
                    }
                }
            }
        }
    }
    
    return legal_moves;
}

bool ChineseChess::has_legal_moves() {
    return !get_legal_moves().empty();
}

void ChineseChess::execute_move(move_type move) {
    int from_pos = move.first;
    int to_pos = move.second;
    int from_x = from_pos / cols;
    int from_y = from_pos % cols;
    int to_x = to_pos / cols;
    int to_y = to_pos % cols;
    
    if (is_illegal(from_x, from_y, to_x, to_y)) {
        throw std::runtime_error("Illegal move");
    }
    
    board[to_x][to_y] = board[from_x][from_y];
    board[from_x][from_y] = EMPTY;
    last_move = move;
    cur_color = -cur_color;
}

std::pair<int, int> ChineseChess::get_game_status() {
    // 检查当前玩家是否被将军
    if (is_check(cur_color)) {
        // 检查是否被将死
        if (!has_legal_moves()) {
            return {1, -cur_color}; // 被将死
        }
        return {0, 0}; // 被将军但未被将死
    } else {
        if (!has_legal_moves()) {
            return {1, 0}; // 困毙，和棋
        }
    }
    
    return {0, 0}; // 游戏继续
}

void ChineseChess::display() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << board[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

std::string ChineseChess::get_piece_symbol(int piece) const {
    if (piece == EMPTY) return "  ";
    
    static const std::map<int, std::string> symbols = {
        // 红方使用简体
        {static_cast<int>(GENERAL) * static_cast<int>(RED), "帅"}, 
        {static_cast<int>(ADVISOR) * static_cast<int>(RED), "士"}, 
        {static_cast<int>(ELEPHANT) * static_cast<int>(RED), "相"},
        {static_cast<int>(HORSE) * static_cast<int>(RED), "马"}, 
        {static_cast<int>(CHARIOT) * static_cast<int>(RED), "车"}, 
        {static_cast<int>(CANNON) * static_cast<int>(RED), "炮"},
        {static_cast<int>(SOLDIER) * static_cast<int>(RED), "兵"},
        // 黑方使用繁体
        {static_cast<int>(GENERAL) * static_cast<int>(GREEN), "將"},
        {static_cast<int>(ADVISOR) * static_cast<int>(GREEN), "仕"}, 
        {static_cast<int>(ELEPHANT) * static_cast<int>(GREEN), "像"},
        {static_cast<int>(HORSE) * static_cast<int>(GREEN), "馬"},
        {static_cast<int>(CHARIOT) * static_cast<int>(GREEN), "車"},
        {static_cast<int>(CANNON) * static_cast<int>(GREEN), "砲"},
        {static_cast<int>(SOLDIER) * static_cast<int>(GREEN), "卒"}
    };
    return symbols.at(piece);
}

void ChineseChess::render() {
    // 获取上一步移动位置
    int last_from = last_move.first;
    int last_to = last_move.second;
    int last_from_x = last_from / cols;
    int last_from_y = last_from % cols;
    int last_to_x = last_to / cols;
    int last_to_y = last_to % cols;

    std::cout << "   ";
    for (int j = 0; j < 9; ++j) {
        std::cout << "  " << j+1 << "  ";
    }
    std::cout << "\n";
    
    std::cout << "  +";
    for (int j = 0; j < 9; ++j) {
        std::cout << "----+";
    }
    std::cout << "\n";
    
    for (int i = 0; i < rows; ++i) {
        std::cout << (char)('A' + i) << " |";
        for (int j = 0; j < 9; ++j) {
            // 标记上一步移动的起点和终点
            if ((i == last_from_x && j == last_from_y) || 
                (i == last_to_x && j == last_to_y)) {
                std::cout << "<" << get_piece_symbol(board[i][j]) << ">";
            } else {
                std::cout << " " << get_piece_symbol(board[i][j]) << " ";
            }
            std::cout << "|";
        }
        std::cout << "\n";
        
        std::cout << "  +";
        for (int j = 0; j < 9; ++j) {
            std::cout << "----+";
        }
        std::cout << "\n";
    }
}