#include "chinese_chess.h"
#include <iostream>
#include <sstream>
#include <algorithm>

std::vector<float> ChineseChess::transorm_board_to_Tensor(board_type board, int to_xy, int cur_player,int no_capture_moves)
{
  auto input_tensor_values = std::vector<float>(ChineseChess::input_size, 0);
  for (int r = 0; r < ChineseChess::rows; r++)
  {
    for (int c = 0; c < ChineseChess::cols; c++)
    {
      int piece = board[r][c];
      if (piece != 0){ // not EMPTY
          int piece_me = piece*cur_color;// 自己的为正，敌人的放chanel小的
          int piece_id = piece_me<0 ? (abs(piece_me)-1) : (piece_me-1 + ChineseChess::piece_num); // chanel 0-6, 7-13
          input_tensor_values[piece_id * ChineseChess::rows * ChineseChess::cols + r * ChineseChess::cols + c] = 1;
      }
      input_tensor_values[ChineseChess::color_channel * ChineseChess::rows * ChineseChess::cols + r * ChineseChess::cols + c] = cur_player;
      if(no_capture_moves >0){
        input_tensor_values[ChineseChess::time_channel * ChineseChess::rows * ChineseChess::cols + r * ChineseChess::cols + c] = float(no_capture_moves)/time_limit;
      }
    }
  }
  if(to_xy >0){
      input_tensor_values[ChineseChess::last_move_channel * ChineseChess::rows * ChineseChess::cols + to_xy] = 1;
  }
  return input_tensor_values;
}


std::vector<float> ChineseChess::transorm_game_to_Tensor()
{
    return ChineseChess::transorm_board_to_Tensor(board, get_last_move(), get_current_color(), no_capture_moves);
}

ChineseChess::ChineseChess(int first_color) : cur_color(first_color), last_move({-1, -1}) {
    init_board();

    // 保存初始状态，便于悔棋
    // board_history.push_back(board);
    // color_history.push_back(cur_color);
}

void ChineseChess::init_board() {
    board = std::vector<std::vector<int>>(rows, std::vector<int>(cols, EMPTY));
    
    // 红方布局 (下方)
    board[9][0] = board[9][8] = static_cast<int>(CHARIOT) * static_cast<int>(FirstColor);
    board[9][1] = board[9][7] = static_cast<int>(HORSE) * static_cast<int>(FirstColor);
    board[9][2] = board[9][6] = static_cast<int>(ELEPHANT) * static_cast<int>(FirstColor);
    board[9][3] = board[9][5] = static_cast<int>(ADVISOR) * static_cast<int>(FirstColor);
    board[9][4] = static_cast<int>(GENERAL) * static_cast<int>(FirstColor);
    board[7][1] = board[7][7] = static_cast<int>(CANNON) * static_cast<int>(FirstColor);
    board[6][0] = board[6][2] = board[6][4] = board[6][6] = board[6][8] = static_cast<int>(SOLDIER) * static_cast<int>(FirstColor);
    // board[6][0] = board[6][2] = board[6][6] = board[6][8] = static_cast<int>(SOLDIER) * static_cast<int>(FirstColor);
    
    // 黑方布局 (上方)
    board[0][0] = board[0][8] = static_cast<int>(CHARIOT) * static_cast<int>(SecondColor);
    board[0][1] = board[0][7] = static_cast<int>(HORSE) * static_cast<int>(SecondColor);
    board[0][2] = board[0][6] = static_cast<int>(ELEPHANT) * static_cast<int>(SecondColor);
    board[0][3] = board[0][5] = static_cast<int>(ADVISOR) * static_cast<int>(SecondColor);
    board[0][4] = static_cast<int>(GENERAL) * static_cast<int>(SecondColor);
    board[2][1] = board[2][7] = static_cast<int>(CANNON) * static_cast<int>(SecondColor);
    board[3][0] = board[3][2] = board[3][4] = board[3][6] = board[3][8] = static_cast<int>(SOLDIER) * static_cast<int>(SecondColor);
    // board[3][0] = board[3][2] = board[3][6] = board[3][8] = static_cast<int>(SOLDIER) * static_cast<int>(SecondColor);
}

// 在 ChineseChess 类中添加这个辅助函数
bool ChineseChess::check_general_face(int from_x, int from_y, int to_x, int to_y) {
    int piece = board[from_x][from_y];
    int color = piece > 0 ? FirstColor : SecondColor;
    int enemy_general = GENERAL * (-color);
    
    // 检查目标位置是否是对方将/帅
    if (board[to_x][to_y] == enemy_general) {
        // 检查将帅之间是否有其他棋子
        int min_x = std::min(from_x, to_x);
        int max_x = std::max(from_x, to_x);
        int y = from_y; // 将帅在同一列
        
        for (int x = min_x + 1; x < max_x; ++x) {
            if (board[x][y] != EMPTY) {
                return false; // 中间有阻挡
            }
        }
        return true; // 将帅直接碰面
    }
    return false;
}

bool ChineseChess::validate_move(int piece, int from_x, int from_y, int to_x, int to_y) {
    int piece_type = abs(piece);
    int color = piece > 0 ? FirstColor : SecondColor;
    
    // 基本检查
    if (to_x < 0 || to_x >= rows || to_y < 0 || to_y >= cols) return false;
    if (board[to_x][to_y] != EMPTY && (board[to_x][to_y] * color > 0)) return false;
    
    int dx = to_x - from_x;
    int dy = to_y - from_y;
    
    switch (piece_type) {
        case GENERAL: {
            // 检查是否是将帅碰面的特殊情况
            if (check_general_face(from_x, from_y, to_x, to_y)) {
                return true; // 允许直接吃掉对方将/帅
            }

            // 只能在九宫内移动
            if (color == FirstColor && (to_x < 7 || to_x > 9 || to_y < 3 || to_y > 5)) return false;
            if (color == SecondColor && (to_x < 0 || to_x > 2 || to_y < 3 || to_y > 5)) return false;
            // 只能走一格直线
            if ((abs(dx) + abs(dy) != 1) || (dx != 0 && dy != 0)) return false;
            return true;
        }
            
        case ADVISOR: {
            // 只能在九宫内移动
            if (color == FirstColor && (to_x < 7 || to_x > 9 || to_y < 3 || to_y > 5)) return false;
            if (color == SecondColor && (to_x < 0 || to_x > 2 || to_y < 3 || to_y > 5)) return false;
            // 只能走一格斜线
            return abs(dx) == 1 && abs(dy) == 1;
        }
            
        case ELEPHANT: {
            // 不能过河
            // std::cout << "ELEPHANT color: " << color << std::endl;
            if (color == FirstColor && to_x < 5) return false;
            if (color == SecondColor && to_x > 4) return false;
            // 走田字且不被蹩脚
            if (abs(dx) != 2 || abs(dy) != 2) return false;
            // std::cout << "ELEPHANT ok " << std::endl;
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
            if (color == FirstColor) {
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

bool ChineseChess::validate_move(int piece, int from_x, int from_y, int mode_code) {
    // int piece_type = abs(piece);
    // int color = piece > 0 ? RED : GREEN;
    
    // 解码mode_code确定目标位置
    int to_x = from_x, to_y = from_y;
    
    if (mode_code < cols) {
        // 横向移动：0=最左，1=左起第二个，...
        to_y = mode_code;
    } else if (mode_code < cols + rows) {
        // 纵向移动：cols=最上，cols+1=从上第二个，...
        to_x = mode_code - cols;
    } else if (mode_code < cols + rows + 4) {
        // 斜向移动1格（士的走法）
        int dir = mode_code - (cols + rows);
        to_x = from_x + (dir < 2 ? -1 : 1);
        to_y = from_y + (dir % 2 == 0 ? -1 : 1);
    } else if (mode_code < cols + rows + 8) {
        // 斜向移动2格（象的走法）
        int dir = mode_code - (cols + rows + 4);
        to_x = from_x + (dir < 4 ? -2 : 2);
        to_y = from_y + (dir % 2 == 0 ? -2 : 2);
    } else {
        // 日字格移动（马的走法）
        int dir = mode_code - (cols + rows + 8);
        if (dir < 4) {
            // 上下方向优先
            to_x = from_x + (dir < 2 ? -2 : 2);
            to_y = from_y + (dir % 2 == 0 ? -1 : 1);
        } else {
            // 左右方向优先
            to_x = from_x + (dir % 2 == 0 ? -1 : 1);
            to_y = from_y + (dir < 6 ? -2 : 2);
        }
    }
    return validate_move(piece, from_x, from_y, to_x, to_y);
}

// 检查当前颜色是否被将军（或者将帅已死亡），现在暂时用不上
// bool ChineseChess::is_check(int color) {
//     // 找到将/帅的位置
//     int general_x = -1, general_y = -1;
//     int general = GENERAL * color;
    
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             if (board[i][j] == general) {
//                 general_x = i;
//                 general_y = j;
//                 break;
//             }
//         }
//         if (general_x != -1) break;
//     }
    
//     if (general_x == -1) return true; // 将/帅被吃掉了
    
//     // 检查是否被将军
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             if (board[i][j] * color < 0) { // 对方棋子
//                 if (validate_move(board[i][j], i, j, general_x, general_y)) {
//                     return true;
//                 }
//             }
//         }
//     }
    
//     return false;
// }

bool ChineseChess::is_illegal(int from_x, int from_y, int to_x, int to_y) {
    if (from_x < 0 || from_x >= rows || from_y < 0 || from_y >= cols) return true;
    if (to_x < 0 || to_x >= rows || to_y < 0 || to_y >= cols) return true;
    
    int piece = board[from_x][from_y];
    if (piece == EMPTY) return true;
    if ((piece > 0 && cur_color == SecondColor) || (piece < 0 && cur_color == FirstColor)) return true;
    
    return !validate_move(piece, from_x, from_y, to_x, to_y);
}

std::vector<int> ChineseChess::get_legal_moves() {
    std::vector<int> legal_moves(this->action_size, 0);
    
    // 1. 首先检查当前方的将/帅是否存在
    bool general_exists = false;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (board[i][j] == GENERAL * cur_color) {
                general_exists = true;
                break;
            }
        }
        if (general_exists) break;
    }
    
    if (!general_exists) {
        return legal_moves; // 将/帅不存在，没有合法走法
    }
    
    // 2. 检查所有当前方棋子的可能移动
    for (int from_x = 0; from_x < rows; ++from_x) {
        for (int from_y = 0; from_y < cols; ++from_y) {
            int piece = board[from_x][from_y];
            
            // 跳过空位和对方棋子
            if (piece == EMPTY || (piece * cur_color < 0)) {
                continue;
            }

            for (int mode_i = 0; mode_i < mode_num; ++mode_i) {
                if (validate_move(piece, from_x, from_y, mode_i)) {
                    legal_moves[(from_x * cols + from_y)*mode_num + mode_i] = 1;
                }
            }
            
            // 3. 检查该棋子的所有可能移动
            // for (int to_x = 0; to_x < rows; ++to_x) {
            //     for (int to_y = 0; to_y < cols; ++to_y) {
            //         if (validate_move(piece, from_x, from_y, to_x, to_y)) {
            //             legal_moves[(from_x * cols + from_y)*cols*rows + to_x * cols + to_y] = 1;
            //         }
            //     }
            // }
        }
    }
    // std::cout << "Legal moves count: " << std::count(legal_moves.begin(), legal_moves.end(), 1) << std::endl;
    
    return legal_moves;
}

bool ChineseChess::has_legal_moves() {
    // 1. 首先检查当前方的将/帅是否存在 TODO:单独写一个函数判断
    bool general_exists = false;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (board[i][j] == GENERAL * cur_color) {
                general_exists = true;
                break;
            }
        }
        if (general_exists) break;
    }
    
    if (!general_exists) {
        return false; // 将/帅不存在，没有合法走法
    }

    // 2. 检查所有当前方棋子的可能移动
    for (int from_x = 0; from_x < rows; ++from_x) {
        for (int from_y = 0; from_y < cols; ++from_y) {
            int piece = board[from_x][from_y];
            
            // 跳过空位和对方棋子
            if (piece == EMPTY || (piece * cur_color < 0)) {
                continue;
            }
            
            // 3. 检查该棋子的所有可能移动
            for (int mode_i = 0; mode_i < mode_num; ++mode_i) {
                if (validate_move(piece, from_x, from_y, mode_i)) {
                    return true;
                }
            }
            // for (int to_x = 0; to_x < rows; ++to_x) {
            //     for (int to_y = 0; to_y < cols; ++to_y) {
            //         if (validate_move(piece, from_x, from_y, to_x, to_y)) {
            //             return true;
            //         }
            //     }
            // }
        }
    }
    return false;
}

int ChineseChess::calculate_move_code(int from_x, int from_y, int to_x, int to_y) {
    int dx = to_x - from_x;
    int dy = to_y - from_y;
    
    // 1. 横向移动（只改变y坐标）
    if (dx == 0 && dy != 0) {
        return to_y; // 直接返回目标列索引
    }
    
    // 2. 纵向移动（只改变x坐标）
    if (dy == 0 && dx != 0) {
        return cols + to_x; // cols + 目标行索引
    }
    
    // 3. 斜向移动1格（士的走法）
    if (abs(dx) == 1 && abs(dy) == 1) {
        int base = cols + rows;
        if (dx == -1 && dy == -1) return base + 0; // 左上
        if (dx == -1 && dy == 1)  return base + 1; // 右上
        if (dx == 1 && dy == -1)  return base + 2; // 左下
        if (dx == 1 && dy == 1)   return base + 3; // 右下
    }
    
    // 4. 斜向移动2格（象的走法）
    if (abs(dx) == 2 && abs(dy) == 2) {
        int base = cols + rows + 4;
        if (dx == -2 && dy == -2) return base + 0; // 左上
        if (dx == -2 && dy == 2)  return base + 1; // 右上
        if (dx == 2 && dy == -2)  return base + 2; // 左下
        if (dx == 2 && dy == 2)   return base + 3; // 右下
    }
    
    // 5. 日字格移动（马的走法）
    if ((abs(dx) == 2 && abs(dy) == 1) || (abs(dx) == 1 && abs(dy) == 2)) {
        int base = cols + rows + 8;
        // 上下方向优先
        if (abs(dx) == 2) {
            if (dx == -2 && dy == -1) return base + 0; // 上左
            if (dx == -2 && dy == 1)  return base + 1; // 上右
            if (dx == 2 && dy == -1)  return base + 2; // 下左
            if (dx == 2 && dy == 1)   return base + 3; // 下右
        }
        // 左右方向优先
        else {
            if (dx == -1 && dy == -2) return base + 4; // 左上
            if (dx == 1 && dy == -2)  return base + 5; // 左下
            if (dx == -1 && dy == 2)  return base + 6; // 右上
            if (dx == 1 && dy == 2)   return base + 7; // 右下
        }
    }
    
    // 非法移动（理论上不应发生，因为调用前应已通过validate_move验证）
    throw std::runtime_error("Invalid move pattern");
}

void ChineseChess::execute_move_by_squeeze_pair(move_type move) {
    int from_pos = move.first;
    int move_code = move.second;
    int from_x = from_pos / cols;
    int from_y = from_pos % cols;
    int to_x = from_x, to_y = from_y;

    // 解码移动方向
    if (move_code < cols) { // 横向移动 0-8
        // 横向移动：0=最左，1=左起第二个，...
        to_y = move_code;
    } else if (move_code < cols + rows) { // 纵向移动 9-18
        // 纵向移动：cols=最上，cols+1=从上第二个，...
        to_x = move_code - cols;
    } else if (move_code < cols + rows + 4) { // 19,20,21,22
        // 斜向移动1格（士的走法）：cols+rows+0=左上，1=右上，2=左下，3=右下
        int dir = move_code - (cols + rows);
        to_x = from_x + (dir < 2 ? -1 : 1);
        to_y = from_y + (dir % 2 == 0 ? -1 : 1);
    } else if (move_code < cols + rows + 8) { // 23,24,25,26
        // 斜向移动2格（象的走法）：cols+rows+4=左上，5=右上，6=左下，7=右下
        int dir = move_code - (cols + rows + 4);
        to_x = from_x + (dir < 2 ? -2 : 2);
        to_y = from_y + (dir % 2 == 0 ? -2 : 2);
        // if (dx == -2 && dy == -2) return base + 0; // 左上
        // if (dx == -2 && dy == 2)  return base + 1; // 右上
        // if (dx == 2 && dy == -2)  return base + 2; // 左下
        // if (dx == 2 && dy == 2)   return base + 3; // 右下
    } else {
        // 日字格移动（马的走法）：cols+rows+8+0=上左，1=上右，2=下左，3=下右，
        // 4=左上，5=左下，6=右上，7=右下
        int dir = move_code - (cols + rows + 8);
        if (dir < 4) {
            // 上下方向优先
            to_x = from_x + (dir < 2 ? -2 : 2);
            to_y = from_y + (dir % 2 == 0 ? -1 : 1);
        } else {
            // 左右方向优先
            to_x = from_x + (dir % 2 == 0 ? -1 : 1);
            to_y = from_y + (dir < 6 ? -2 : 2);
        }
    }

    // 检查移动是否合法
    // std::cout << "is right? from_x: " << from_x << ", from_y: " << from_y << ", to_x: " << to_x << ", to_y: " << to_y << std::endl;
    if (is_illegal(from_x, from_y, to_x, to_y)) {
        throw std::runtime_error("Illegal move");
    }

    // 检查是否是吃子/检查步数
    // if (board[to_x][to_y] != EMPTY) {
    //     no_capture_moves = 0;
    // } else {
        no_capture_moves++;
    // }

    // // 保存当前状态到历史记录 [模拟时没必要干这个！！]
    // board_history.push_back(board);
    // color_history.push_back(cur_color);

    // 执行移动
    board[to_x][to_y] = board[from_x][from_y];
    board[from_x][from_y] = EMPTY;
    last_move = {from_pos, to_x * cols + to_y};
    cur_color = -cur_color;
}

void ChineseChess::execute_move(int move) {
    int from_pos = move / (this->mode_num);
    int move_mode = move % (this->mode_num);
    execute_move_by_squeeze_pair({from_pos,move_mode});
}

// 添加悔棋函数
bool ChineseChess::undo_move() {
    if (board_history.size() <= 1) {
        return false; // 无法悔棋
    }
    
    // 移除当前状态
    board_history.pop_back();
    color_history.pop_back();
    
    // 恢复上一状态
    board = board_history.back();
    cur_color = color_history.back();
    
    // 重置未吃子步数（这里简化处理，实际可能需要更复杂的逻辑）
    no_capture_moves = 0;
    
    return true;
}

std::pair<int, int> ChineseChess::get_game_status() {
    // 检查xx步未吃子和棋规则
    if (no_capture_moves >= time_limit) {
        return {1, 0}; // xx回合不吃子算和棋（按国际象棋规则来，避免没有和棋，永无止尽走下去）
    }
    if (!has_legal_moves()) { //一般是因为将帅已死亡，或者没有棋子可动（将帅也被自己的棋子困死）
        return {1, -cur_color};
    }
    
    // // 检查当前玩家是否被将军
    // if (is_check(cur_color)) {
    //     // 检查是否被将死
    //     if (!has_legal_moves()) {
    //         return {1, -cur_color}; // 被将死
    //     }
    //     return {0, 0}; // 被将军但未被将死
    // } else {
    //     // 检查是否困毙
    //     if (!has_legal_moves()) {
    //         return {1, -cur_color}; // 困毙，当前玩家输
    //     }
    // }
    
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
        {static_cast<int>(GENERAL) * static_cast<int>(FirstColor), "帅"}, 
        {static_cast<int>(ADVISOR) * static_cast<int>(FirstColor), "士"}, 
        {static_cast<int>(ELEPHANT) * static_cast<int>(FirstColor), "相"},
        {static_cast<int>(HORSE) * static_cast<int>(FirstColor), "马"}, 
        {static_cast<int>(CHARIOT) * static_cast<int>(FirstColor), "车"}, 
        {static_cast<int>(CANNON) * static_cast<int>(FirstColor), "炮"},
        {static_cast<int>(SOLDIER) * static_cast<int>(FirstColor), "兵"},
        // 黑方使用繁体+古称（卒）+单人旁汉字
        {static_cast<int>(GENERAL) * static_cast<int>(SecondColor), "將"},
        {static_cast<int>(ADVISOR) * static_cast<int>(SecondColor), "仕"}, 
        {static_cast<int>(ELEPHANT) * static_cast<int>(SecondColor), "像"},
        {static_cast<int>(HORSE) * static_cast<int>(SecondColor), "馬"},
        {static_cast<int>(CHARIOT) * static_cast<int>(SecondColor), "車"},
        {static_cast<int>(CANNON) * static_cast<int>(SecondColor), "砲"},
        {static_cast<int>(SOLDIER) * static_cast<int>(SecondColor), "卒"}
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
            if (i == 4){
                std::cout << "oooo+";
            }else{
                std::cout << "----+";
            }
            
        }
        std::cout << "\n";
    }
    std::cout << "无吃子步数："<< no_capture_moves  << '/' << time_limit << std::endl;
}