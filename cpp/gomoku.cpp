// #include <math.h>
#include <iostream>
#include <sstream>

#include <gomoku.h>

Gomoku::Gomoku(const unsigned int n, const unsigned int n_in_row, int first_color)
    : n(n), n_in_row(n_in_row), cur_color(first_color), last_move(-1) {
  this->board = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));
  this->record_list.clear();
}

bool Gomoku::is_illegal(unsigned int x, unsigned int y){
  return x > this->n-1 || y>this->n-1 || this->board[x][y] != 0;
} 

std::vector<int> Gomoku::get_legal_moves() {
  auto n = this->n;
  std::vector<int> legal_moves(this->get_action_size(), 0);

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        legal_moves[i * n + j] = 1;
      }
    }
  }

  return legal_moves;
}

bool Gomoku::has_legal_moves() {
  auto n = this->n;

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        return true;
      }
    }
  }
  return false;
}

void Gomoku::execute_move(move_type move) {
  auto i = move / this->n;
  auto j = move % this->n;

  if (this->board[i][j] != 0) {
    throw std::runtime_error("execute_move borad[i][j] != 0.");
  }

  this->board[i][j] = this->cur_color;
  this->record_list.push_back(move);
  this->last_move = move;
  // change player
  this->cur_color = -this->cur_color;
}

void Gomoku::take_back_move()
{
  size_t s_tmp = this->record_list.size();
  if (s_tmp >= 1)
    this->record_list.pop_back();
  size_t s_new = this->record_list.size();

  if (s_new + 1 == s_tmp)
  {
    auto i = this->last_move / this->n;
    auto j = this->last_move % this->n;
    this->board[i][j] = 0;
    this->last_move = this->record_list.back();
    // change player
    this->cur_color = -this->cur_color;
  }
}

std::pair<int, int> Gomoku::get_game_status() {
  // return (is ended, winner)
  auto n = this->n;
  auto n_in_row = this->n_in_row;

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        continue;
      }

      if (j <= n - n_in_row) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i][j + k];
        }
        if (abs(sum) == n_in_row) {
          return {1, this->board[i][j]};
        }
      }

      if (i <= n - n_in_row) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i + k][j];
        }
        if (abs(sum) == n_in_row) {
          return {1, this->board[i][j]};
        }
      }

      if (i <= n - n_in_row && j <= n - n_in_row) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i + k][j + k];
        }
        if (abs(sum) == n_in_row) {
          return {1, this->board[i][j]};
        }
      }

      if (i <= n - n_in_row && j >= n_in_row - 1) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i + k][j - k];
        }
        if (abs(sum) == n_in_row) {
          return {1, this->board[i][j]};
        }
      }
    }
  }

  if (this->has_legal_moves()) {
    return {0, 0};
  } else {
    return {1, 0};
  }
}

void Gomoku::display() const {
  auto n = this->board.size();

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      std::cout << this->board[i][j] << ", ";
    }
    std::cout << std::endl;
  }
}

void Gomoku::render() {
    std::ostringstream out;
    const size_t n = this->board.size();
    //out << "step ?" << std::endl;

    const size_t cell_size = 3;
    const size_t row_size = (cell_size + 1) * n + 1;
    //std::vector<char> line[row_size + 1];
    char* line = new char[row_size + 1];
    char* line2 = new char[row_size + 1];
    for (int i = 0; i < row_size; i++) {
      if (i % (cell_size + 1) == 0)
        line[i] = '+';
      else
        line[i] = '-';

      line2[i] = ' ';
    }
    line[row_size] = 0;
    line2[row_size] = 0;
    out << line << std::endl;
    unsigned int this_i = this->last_move / this->n;
    unsigned int this_j = this->last_move % this->n;
    for (unsigned int i = 0; i < n; i++) {
      for (unsigned int j = 0; j < n; j++) {
        line2[j * (cell_size + 1)] = '|';
        if (this_i == i) {
        	if (this_j == j) {
        		line2[j * (cell_size + 1)] = '[';
        	}
        	else if (this_j == j - 1) {
        		line2[j * (cell_size + 1)] = ']';
        	}
        }

        int st = j * (cell_size + 1) + cell_size / 2 + 1;
        if (board[i][j] == 1)
          line2[st] = '#';
        else if (board[i][j] == -1)
          line2[st] = 'O';
        else
          line2[st] = ' ';
      }
      line2[row_size - 1] = '|';
      if (this_i == i && this_j == this->n - 1) {
      	line2[row_size - 1] = ']';
      }

      out << line2 << " " << (char)('A' + i) << std::endl;
      out << line << std::endl;
  }
  //char column_no[row_size] = {0};
    char* column_no = new char[row_size] {0};
	unsigned int offset = 0;
	for (unsigned int i = 1; i <= n; i++) {
		offset += snprintf(column_no + offset, row_size - offset, "  %-2d", i); //snprintf -> sprintf_s
	}
	out << column_no;
	puts(out.str().c_str());
}

