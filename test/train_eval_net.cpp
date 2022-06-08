#include <iostream>
#include <fstream>
#include <sstream> 

#include <mcts.h>

#include <onnx.h>
#include <play.h>
#include <common.h>

using namespace std;

void generate_data_for_train(int current_weight, int start_batch_id) {
    string path = "./weights/" + to_string(current_weight) + ".onnx";

    NeuralNetwork* model = new NeuralNetwork(path, NUM_MCT_THREADS * NUM_MCT_SIMS);
    SelfPlay* sp = new SelfPlay(model);
    sp->self_play_for_train(NUM_TRAIN_THREADS, start_batch_id);
}

void play_for_eval(NeuralNetwork *a, NeuralNetwork *b, bool a_first, int *win_table, bool do_render, const int a_mct_sims, const int b_mct_sims)
{
    MCTS ma(a, NUM_MCT_THREADS, C_PUCT, a_mct_sims, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
    MCTS mb(b, NUM_MCT_THREADS, C_PUCT, b_mct_sims, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
    int step = 0;
    auto g = std::make_shared<Gomoku>(BORAD_SIZE, N_IN_ROW, BLACK);
    std::pair<int, int> game_state = g->get_game_status();
    // std::cout << episode << " th game!!" << std::endl;
    while (game_state.first == 0)
    {
        int res = (step + a_first) % 2 ? ma.get_best_action(g.get()) : mb.get_best_action(g.get());
        ma.update_with_move(res);
        mb.update_with_move(res);
        g->execute_move(res);
        if (do_render)
            g->render();
        game_state = g->get_game_status();
        step++;
    }
    cout << "eval: total step num = " << step << endl;

    if ((game_state.second == BLACK && a_first) || (game_state.second == WHITE && !a_first))
    {
        cout << "winner = a" << endl;
        win_table[0]++;
    }
    else if ((game_state.second == BLACK && !a_first) || (game_state.second == WHITE && a_first))
    {
        cout << "winner = b" << endl;
        win_table[1]++;
    }
    else if (game_state.second == 0)
        win_table[2]++;
}

vector<int> eval(int weight_a, int weight_b, unsigned int game_num,int a_sims,int b_sims) {
    int win_table[3] = { 0,0,0 };

    ThreadPool *thread_pool = new ThreadPool(game_num);
    NeuralNetwork* nn_a = nullptr;
    NeuralNetwork* nn_b = nullptr;

    if (weight_a >= 0) {
        string path = "./weights/" + to_string(weight_a) + ".onnx";
        nn_a = new NeuralNetwork(path, game_num * a_sims);
        cout << "NeuralNetwork A load: " << weight_a << endl;
    }
    else {
        cout << "NeuralNetwork A applies random policy!" << endl;
    }

    if (weight_b >= 0) {
        string path = "./weights/" + to_string(weight_b) + ".onnx";
        nn_b = new NeuralNetwork(path, game_num * b_sims);
        cout << "NeuralNetwork B load: " << weight_b << endl;
    }
    else {
        cout << "NeuralNetwork B applies random policy!" << endl;
    }

    std::vector<std::future<void>> futures;
    //NeuralNetwork* a = new NeuralNetwork(NUM_MCT_THREADS * NUM_MCT_SIMS);
    for (unsigned int i = 0; i < game_num; i++) {
        auto future = thread_pool->commit(std::bind(play_for_eval, nn_a, nn_b, i % 2 == 0, win_table,i==0, a_sims, b_sims));
        futures.emplace_back(std::move(future));
    }
    for (unsigned int i = 0; i < futures.size(); i++) {
        futures[i].wait();
        if (nn_a != nullptr){
            nn_a->batch_size = max((unsigned)1, (game_num - i) * NUM_MCT_THREADS);
        }
        if (nn_b != nullptr){
            nn_b->batch_size = max((unsigned)1, (game_num - i) * NUM_MCT_THREADS);
        }

    }
    //cout << "win_table = " << win_table[0] << win_table[1] << win_table [2] << endl;

    return { win_table[0], win_table[1],win_table[2] };
}

int main(int argc, char *argv[])
{
    if (strcmp(argv[1], "prepare") == 0)
    {
        cout << "Prepare for training." << endl;
        //system("mkdir weights");
#ifdef _WIN32
        system("mkdir .\\weights");
        system("mkdir .\\data");
#elif __linux__
        auto ret = system("mkdir ./weights");
        // cout <<ret <<endl;
        ret = system("mkdir ./data");
#endif
        ofstream weight_logger_writer("current_and_best_weight.txt");
        weight_logger_writer << 0 << " " << 0;
        weight_logger_writer.close();

        ofstream random_mcts_logger_writer("random_mcts_number.txt");
        random_mcts_logger_writer << NUM_MCT_SIMS;
        random_mcts_logger_writer.close();

        cout << "Next: Generate initial weight by python." << endl;
        //system("python ..\\python\\learner.py");
    }else if (strcmp(argv[1], "generate") == 0) {
        cout << "generate " << atoi(argv[2])  << "-th batch."<< endl;
        int current_weight;
        int best_weight;

        ifstream logger_reader("current_and_best_weight.txt");
        logger_reader >> current_weight;
        //logger_reader >> best_weight;
        if (current_weight < 0) {
            cout << "LOAD error,check current_and_best_weight.txt" << endl;
            return -1;
        }
        // logger_reader >> temp[1];
        logger_reader.close();
        cout << "Generating... current_weight = " << current_weight << endl;
        generate_data_for_train(current_weight, atoi(argv[2]) * NUM_TRAIN_THREADS);
    }else if (strcmp(argv[1], "eval") == 0) {
		int current_weight;
        int best_weight;

        ifstream weight_logger_reader("current_and_best_weight.txt");
        weight_logger_reader >> current_weight;
        weight_logger_reader >> best_weight;
        cout << "Evaluating... current_weight = " << current_weight << " and best_weight = " << best_weight << endl;

        int game_num = atoi(argv[2]);

        auto result = eval(current_weight, best_weight, game_num, NUM_MCT_SIMS, NUM_MCT_SIMS);
        string result_log_info = to_string(current_weight) + "-th weight win: " + to_string(result[0]) + "  " + to_string(best_weight) + "-th weight win: " + to_string(result[1]) + "  tie: " + to_string(result[2]) + "\n";

        float win_ratio = result[0] / (result[1]+0.01);
        if (win_ratio > 1.2 ) {
            result_log_info += "new best weight: " + to_string(current_weight) + " generated!!!!\n";
            ofstream weight_logger_writer("current_and_best_weight.txt");
            weight_logger_writer << current_weight << " " << current_weight;
            weight_logger_writer.close();
        }
        cout << result_log_info;

        int random_mcts_simulation;
        ifstream random_mcts_logger_reader("random_mcts_number.txt");
        random_mcts_logger_reader >> random_mcts_simulation;

        int nn_mcts_simulation = NUM_MCT_SIMS / 16; // can not be too small !!

        vector<int> result_random_mcts = eval(current_weight, -1, game_num, nn_mcts_simulation, random_mcts_simulation);
        
        
        string result_log_info2 = to_string(current_weight) + "-th weight with mcts ["+ to_string(nn_mcts_simulation) + "] win: " + to_string(result_random_mcts[0]) + "  Random mcts ["+to_string(random_mcts_simulation)+ "] win: " + to_string(result_random_mcts[1]) + "  tie: " + to_string(result_random_mcts[2]) + "\n";
        if (result_random_mcts[0] - game_num==0 && random_mcts_simulation < 5000) {
            random_mcts_simulation += 100;
            result_log_info2 += "add random mcts number to: " + to_string(random_mcts_simulation) + "\n";

            ofstream random_mcts_logger_writer("random_mcts_number.txt");
            random_mcts_logger_writer << random_mcts_simulation;
            random_mcts_logger_writer.close();
        }
        cout << result_log_info2;

        ofstream detail_logger_writer("logger.txt", ios::app);
        detail_logger_writer << result_log_info << result_log_info2;
        detail_logger_writer.close();
    }else{
        cout << "Do nothing...check your input!!" << endl;
    }
}
