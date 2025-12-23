#include "go.h"
#include <iostream>
#include <sstream>
#include <queue>
#include <algorithm>
#include <random>
#include <cmath>
#include <iomanip>  // 用于std::setprecision

Go::Go(unsigned int width, unsigned int height, int first_color)
    : width(width), height(height), cur_color(first_color),
      last_move(-1), black_captures(0), white_captures(0),
      current_hash(0), game_ended(false),ko_move(-1) {
    
    board = std::vector<std::vector<int>>(height, std::vector<int>(width, 0));

    // 然后计算动态贴目和最终得分
    double area_ratio = (double)(width * height) / (19.0 * 19.0);
    komi = std::max(0.0, std::min(7.5, 7.5 * area_ratio));
    
    // 初始化Zobrist哈希表
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(1, UINT64_MAX);
    
    zobrist_black.resize(height, std::vector<uint64_t>(width));
    zobrist_white.resize(height, std::vector<uint64_t>(width));
    
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            zobrist_black[y][x] = dis(gen);
            zobrist_white[y][x] = dis(gen);
        }
    }
    
    position_history.insert(current_hash);
}

// 寻找棋子组
Go::Group Go::find_group(unsigned int x, unsigned int y) {
    Group group;
    int color = board[y][x];
    if (color == Empty) return group;
    
    std::set<std::pair<unsigned int, unsigned int>> visited;
    std::queue<std::pair<unsigned int, unsigned int>> q;
    q.push({x, y});
    visited.insert({x, y});
    
    const int dx[] = {-1, 1, 0, 0};
    const int dy[] = {0, 0, -1, 1};
    
    while (!q.empty()) {
        auto [cx, cy] = q.front(); q.pop();
        group.stones.insert({cx, cy});
        
        for (int i = 0; i < 4; i++) {
            int nx = cx + dx[i];
            int ny = cy + dy[i];
            
            if (nx < 0 || nx >= (int)width || ny < 0 || ny >= (int)height) continue;
            
            if (board[ny][nx] == color && !visited.count({nx, ny})) {
                visited.insert({nx, ny});
                q.push({nx, ny});
            } else if (board[ny][nx] == Empty) {
                group.liberties.insert({nx, ny});
            }
        }
    }
    
    return group;
}

int Go::count_liberties(unsigned int x, unsigned int y) {
    Group group = find_group(x, y);
    return group.liberties.size();
}

void Go::remove_group(unsigned int x, unsigned int y) {
    Group group = find_group(x, y);
    for (auto [sx, sy] : group.stones) {
        board[sy][sx] = Empty;
        current_hash ^= (cur_color == FirstColor) ? zobrist_white[sy][sx] : zobrist_black[sy][sx];
    }
}

// 自杀判断
bool Go::is_suicide(unsigned int x, unsigned int y, int color) {
    board[y][x] = color;
    bool has_liberty = false;
    
    const int dx[] = {-1, 1, 0, 0};
    const int dy[] = {0, 0, -1, 1};
    
    // 检查是否能提对方棋
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx < 0 || nx >= (int)width || ny < 0 || ny >= (int)height) continue;
        
        if (board[ny][nx] == -color && count_liberties(nx, ny) == 0) {
            board[y][x] = Empty;
            return false; // 能提子，不是自杀
        }
    }
    
    // 检查自己的气
    if (count_liberties(x, y) > 0) {
        board[y][x] = Empty;
        return false;
    }
    
    board[y][x] = Empty;
    return true;
}

// 劫判断（使用历史哈希）
bool Go::violates_ko(unsigned int x, unsigned int y, int color) {
    uint64_t temp_hash = current_hash;
    temp_hash ^= (color == FirstColor) ? zobrist_black[y][x] : zobrist_white[y][x];
    
    // 模拟提子后的局面
    const int dx[] = {-1, 1, 0, 0};
    const int dy[] = {0, 0, -1, 1};
    
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx < 0 || nx >= (int)width || ny < 0 || ny >= (int)height) continue;
        
        if (board[ny][nx] == -color && count_liberties(nx, ny) == 1) {
            // 模拟提子
            Group group = find_group(nx, ny);
            for (auto [sx, sy] : group.stones) {
                temp_hash ^= (color == FirstColor) ? zobrist_white[sy][sx] : zobrist_black[sy][sx];
            }
        }
    }
    
    return position_history.find(temp_hash) != position_history.end();
}

bool Go::is_illegal(unsigned int x, unsigned int y) {
    if (x >= width || y >= height) return true;
    if (board[y][x] != Empty) return true;
    if (is_suicide(x, y, cur_color)) return true;
    if (violates_ko(x, y, cur_color)) return true;
    return false;
}

void Go::execute_move(move_type move) {
    if (move == -1) { // pass
        cur_color = -cur_color;
        last_move = -1;
        return;
    }
    
    unsigned int x = move % width;
    unsigned int y = move / width;
    
    if (is_illegal(x, y)) {
        print_illegal_reason(x, y);
        throw std::runtime_error("非法着法");
    }
    
    // 落子并更新哈希
    board[y][x] = cur_color;
    current_hash ^= (cur_color == FirstColor) ? zobrist_black[y][x] : zobrist_white[y][x];
    
    const int dx[] = {-1, 1, 0, 0};
    const int dy[] = {0, 0, -1, 1};
    
    // 提子
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        if (nx < 0 || nx >= (int)width || ny < 0 || ny >= (int)height) continue;
        
        if (board[ny][nx] == -cur_color && count_liberties(nx, ny) == 0) {
            Group group = find_group(nx, ny);
            int count = group.stones.size();
            remove_group(nx, ny);
            
            if (cur_color == FirstColor) black_captures += count;
            else white_captures += count;
        }
    }
    
    last_move = move;
    cur_color = -cur_color;
    
    // 记录新局面
    position_history.insert(current_hash);
}

// 洪水填充计算领地
std::pair<int, int> Go::count_territory() {
    std::set<std::pair<unsigned int, unsigned int>> visited;
    std::set<std::pair<unsigned int, unsigned int>> black_territory;
    std::set<std::pair<unsigned int, unsigned int>> white_territory;
    
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            if (board[y][x] == Empty && !visited.count({x, y})) {
                std::set<std::pair<unsigned int, unsigned int>> region;
                std::set<int> adjacent_colors;
                flood_fill_territory(x, y, visited, region, adjacent_colors);
                
                if (adjacent_colors.size() == 1) {
                    if (*adjacent_colors.begin() == FirstColor) {
                        black_territory.insert(region.begin(), region.end());
                    } else {
                        white_territory.insert(region.begin(), region.end());
                    }
                }
            }
        }
    }
    
    return {black_territory.size(), white_territory.size()};
}

void Go::flood_fill_territory(unsigned int x, unsigned int y,
                              std::set<std::pair<unsigned int, unsigned int>>& visited,
                              std::set<std::pair<unsigned int, unsigned int>>& region,
                              std::set<int>& adjacent_colors) {
    std::queue<std::pair<unsigned int, unsigned int>> q;
    q.push({x, y});
    visited.insert({x, y});
    region.insert({x, y});
    
    const int dx[] = {-1, 1, 0, 0};
    const int dy[] = {0, 0, -1, 1};
    
    while (!q.empty()) {
        auto [cx, cy] = q.front(); q.pop();
        
        for (int i = 0; i < 4; i++) {
            int nx = cx + dx[i];
            int ny = cy + dy[i];
            
            if (nx < 0 || nx >= (int)width || ny < 0 || ny >= (int)height) continue;
            
            if (board[ny][nx] == Empty && !visited.count({nx, ny})) {
                visited.insert({nx, ny});
                region.insert({nx, ny});
                q.push({nx, ny});
            } else if (board[ny][nx] != Empty) {
                adjacent_colors.insert(board[ny][nx]);
            }
        }
    }
}

// 自动终局检测 + 智能计分
std::pair<int, int> Go::get_game_status() {
    // 检查是否还有合法着法
    if (!has_legal_moves()) {
        game_ended = true;
    }
    
    // 自动终局：双方无有效着法或棋盘填满
    if (game_ended || position_history.size() > time_limit) {
        // 在get_game_status()函数中找到领地计算后的部分
        auto [black_territory, white_territory] = count_territory();

        // 计算棋盘上的剩余棋子
        int black_stones = 0, white_stones = 0;
        for (unsigned int y = 0; y < height; y++) {
            for (unsigned int x = 0; x < width; x++) {
                if (board[y][x] == FirstColor) black_stones++;
                else if (board[y][x] == SecondColor) white_stones++;
            }
        }



        // 中国规则：子+目+俘虏
        double black_score = black_captures + black_territory + black_stones;
        double white_score = white_captures + white_territory + white_stones + komi;

        
        for (unsigned int y = 0; y < height; y++) {
            for (unsigned int x = 0; x < width; x++) {
                if (board[y][x] == FirstColor) black_score++;
                else if (board[y][x] == SecondColor) white_score++;
            }
        }
        
        int winner = (black_score > white_score) ? FirstColor : SecondColor;

        // std::cout << "\n=== 终局计分 ===" << std::endl;
        // std::cout << "黑方得分: " << black_score 
        //         << " (提子:" << black_captures 
        //         << " 领地:" << black_territory 
        //         << " 棋子:" << black_stones << ")" << std::endl;
        // std::cout << "白方得分: " << white_score 
        //         << " (提子:" << white_captures 
        //         << " 领地:" << white_territory 
        //         << " 棋子:" << white_stones 
        //         << " 贴目:" << komi << ")" << std::endl;
        // std::cout << "胜负: " << (winner == Black ? "黑胜" : "白胜") << std::endl;
        return {1, winner};
    }
    
    return {0, 0};
}

// 其他方法实现...
bool Go::has_legal_moves() {
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            if (!is_illegal(x, y)) return true;
        }
    }
    return false;
}

std::vector<int> Go::get_legal_moves() {
    std::vector<int> legal_moves(this->get_action_size(), 0);
    
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (board[y][x] == 0 && !is_illegal(x, y)) {  // 双重检查
                legal_moves[idx] = 1;  // ✅ 赋值为1！
            }
        }
    }
    return legal_moves;  // 正确返回掩码
}

void Go::render() {
    std::ostringstream out;
    const size_t cell_size = 3;
    const size_t row_size = (cell_size + 1) * width + 1;
    
    char* line = new char[row_size + 1];
    char* line2 = new char[row_size + 1];
    
    for (size_t i = 0; i < row_size; i++) {
        if (i % (cell_size + 1) == 0) line[i] = '+';
        else line[i] = '-';
        line2[i] = ' ';
    }
    line[row_size] = line2[row_size] = '\0';
    
    out << line << std::endl;
    
    unsigned int last_y = (last_move == -1) ? -1 : last_move / width;
    unsigned int last_x = (last_move == -1) ? -1 : last_move % width;
    
    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            line2[x * (cell_size + 1)] = '|';
            
            if (last_move != -1 && last_y == y && last_x == x) {
                line2[x * (cell_size + 1)] = '[';
            } else if (last_move != -1 && last_y == y && last_x == x - 1) {
                line2[x * (cell_size + 1)] = ']';
            }
            
            int st = x * (cell_size + 1) + cell_size / 2 + 1;
            if (board[y][x] == FirstColor) line2[st] = '#';
            else if (board[y][x] == SecondColor) line2[st] = 'O';
            else line2[st] = ' ';
        }
        
        line2[row_size - 1] = '|';
        if (last_move != -1 && last_y == y && last_x == width - 1) {
            line2[row_size - 1] = ']';
        }
        
        out << line2 << " " << (char)('A' + y) << std::endl;
        out << line << std::endl;
    }
    
    char* column_no = new char[row_size];
    int offset = 0;
    for (unsigned int x = 1; x <= width; x++) {
        offset += snprintf(column_no + offset, row_size - offset, "  %-2d", x);
    }
    out << column_no;
    
    // out << "\n\n黑提:" << black_captures 
    //     << " 白提:" << white_captures << "(贴7.5目)";
    out << "\n\n黑提:" << black_captures 
        << " 白提:" << white_captures 
        << "(贴" << std::fixed << std::setprecision(1) << komi << "目)";

    
    puts(out.str().c_str());
}

void Go::print_illegal_reason(unsigned int x, unsigned int y) {
    if (x >= width || y >= height)
        {std::cout<< "位置太大, x = " << x <<", y =" << y << std::endl;}
    else if (board[y][x] != Empty)
        {std::cout<< "非空" <<std::endl;}
    else if (is_suicide(x, y, cur_color))
        {std::cout<< "自杀" <<std::endl;}
    else if (violates_ko(x, y, cur_color))
        {std::cout<< "劫" <<std::endl;}
}

// std::vector<float> transorm_board_to_Tensor(board_type board, int last_move, int cur_player)
// {
//   auto input_tensor_values = std::vector<float>(CHANNEL_SIZE * this->width * this->height);
//   int first = 0;
//   int second = 0;
//   if (cur_player == FirstColor)
//   {
//     second = 1; //Black currently play = All black positions occupy the 0-th dimension in board
//   }
//   else
//   {
//     first = 1;
//   }
//   for (int r = 0; r < BORAD_SIZE; r++)
//   {
//     for (int c = 0; c < BORAD_SIZE; c++)
//     {
//       switch (board[r][c])
//       {
//       case 1:
//         input_tensor_values[first * BORAD_SIZE * BORAD_SIZE + r * BORAD_SIZE + c] = 1;
//         break;
//       case -1:
//         input_tensor_values[second * BORAD_SIZE * BORAD_SIZE + r * BORAD_SIZE + c] = 1;
//         break;
//       default:
//         break;
//       }
//     }
//     if(last_move >=0){
//       input_tensor_values[2 * BORAD_SIZE * BORAD_SIZE + last_move] = 1;
//     }
//   }
//   return input_tensor_values;
// }

// std::vector<float> transorm_game_to_Tensor()
// {
//   return transorm_board_to_Tensor(get_board(), get_last_move(), get_current_color());
// }

std::vector<float> Go::transorm_board_to_Tensor(board_type board, int last_xy, int cur_player, int step)
{
  auto input_tensor_values = std::vector<float>(get_input_size(), 0);
  for (int r = 0; r < height; r++)
  {
    for (int c = 0; c < width; c++)
    {
      int piece = board[r][c];
      if (piece != Empty){ // not EMPTY
        int piece_me = piece*cur_player;// 自己的为正，敌人的放chanel小的
        int piece_id = piece_me<0 ? (abs(piece_me)-1) : (piece_me-1 + Go::piece_num); // chanel 0-6, 7-13
        input_tensor_values[piece_id * height * width + r * width + c] = 1;
      }
      input_tensor_values[Go::color_channel * height * width + r * width + c] = cur_player;
      if(step >0){
        input_tensor_values[Go::time_channel * height * width + r * width + c] = float(step)/time_limit;
      }
    }
  }
  if(last_xy >0){
      input_tensor_values[Go::last_move_channel * height * width + last_xy] = 1;
  }
  return input_tensor_values;
}


std::vector<float> Go::transorm_game_to_Tensor()
{
    return transorm_board_to_Tensor(get_board(), get_last_move(), get_current_color(), get_step());
}