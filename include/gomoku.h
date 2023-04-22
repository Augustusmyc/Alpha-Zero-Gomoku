#pragma once

#include <tuple>
#include <vector>
#include <common.h>

//using namespace customType;

class Gomoku {
public:
  using move_type = int;

  Gomoku(const unsigned int n, const unsigned int n_in_row, int first_color);

  bool has_legal_moves();
  std::vector<int> get_legal_moves();
  void execute_move(move_type move);
  void take_back_move();
  std::pair<int, int> get_game_status();
  void display() const;
  void render();
  //bool is_illegal(move_type move);
  bool is_illegal(unsigned int x,unsigned int y);

  inline unsigned int get_action_size() const { return this->n * this->n; }
  inline board_type get_board() const { return this->board; }
  inline move_type get_last_move() const { return this->last_move; }
  inline int get_current_color() const { return this->cur_color; }
  inline unsigned int get_n() const { return this->n; }

private:
  board_type board;      // game borad
  std::vector<move_type> record_list;   // record moves in order
  const unsigned int n;        // board size
  const unsigned int n_in_row; // 5 in row or else

  int cur_color;       // current player's color
  move_type last_move; // last move
};
