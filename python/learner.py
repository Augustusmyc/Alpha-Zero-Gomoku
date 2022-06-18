from collections import deque
from os import path, mkdir
import os
import torch
import numpy as np
import common as config
# import pickle
# import concurrent.futures
import random, struct
from functools import reduce

import sys

sys.path.append('../build')

from neural_network import NeuralNetWorkWrapper


class Learner():
    def __init__(self, config):
        # gomoku
        self.n = config['n']
        self.n_in_row = config['n_in_row']
        # self.gomoku_gui = GomokuGUI(config['n'], config['human_color'])
        self.action_size = config['action_size']

        # train
        self.num_iters = config['num_iters']
        self.num_eps = config['num_eps']
        self.num_train_threads = config['num_train_threads']
        self.check_freq = config['check_freq']
        self.num_contest = config['num_contest']
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.temp = config['temp']
        self.update_threshold = config['update_threshold']
        self.num_explore = config['num_explore']

        self.examples_buffer = deque([], maxlen=config['examples_buffer_max_len'])

        use_GPU = torch.cuda.is_available()

        # neural network
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.nnet = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
                                         config['num_channels'], config['n'], 
                                         self.action_size, config['input_channel_size'])

        # start gui
        # t = threading.Thread(target=self.gomoku_gui.loop)
        # t.start()

    def learn(self, model_dir,model_id):
        # train the model by self play

        model_path = path.join(model_dir, str(model_id))
        assert path.exists(model_path+'.pkl'),f"{model_path+'.pkl'} does not exist!!!"
        print(f"loading {model_id}-th model")
        self.nnet.load_model(model_path)

        # model_id = 0
        # if model_dir==None:
        #     print("debug mode: best_model_dir = join('..','build','weights', str(model_id))")
        #     model_dir = path.join('..','build','weights')
        # model_path = path.join(model_dir, str(model_id))
        # if path.exists(model_path+'.pkl'):
        #     print(f"loading {model_id}-th model")
        #     self.nnet.load_model(model_path)
        #     #self.load_samples()
        # else:
        #     print("prepare: save 0-th model")
        #     # save torchscript
        #     # self.nnet.save_model()
        #     self.nnet.save_model(model_path)

        data_path = path.join('..', 'build', 'data')
        train_data = self.load_samples(data_path)
        random.shuffle(train_data)

        # train neural network
        epochs = self.epochs * (len(train_data) + self.batch_size - 1) // self.batch_size
        self.nnet.train(train_data, min(self.batch_size,len(train_data)), int(epochs))

        model_path = path.join(model_dir, str(model_id+1))
        self.nnet.save_model(model_path)
        torch.cuda.empty_cache()

    def get_symmetries(self, board, pi, last_action):
        # mirror, rotational
        assert (len(pi) == self.action_size)  # 1 for pass

        pi_board = np.reshape(pi, (self.n, self.n))
        last_action_board = np.zeros((self.n, self.n))
        if(last_action != -1):
            last_action_board[last_action // self.n][last_action % self.n] = 1
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                newAction = np.rot90(last_action_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                    newAction = np.fliplr(last_action_board)
                l += [(newB, newPi.ravel(), np.argmax(newAction) if last_action != -1 else -1)]
        return l

    def load_samples(self, folder):
        """load self.examples_buffer
        """
        BOARD_SIZE = self.n
        train_examples = []
        data_files = os.listdir(folder)
        for file_name in data_files:
            file_path = path.join(folder, file_name)
            with open(file_path, 'rb') as binfile:
                # size = os.path.getsize(filepath) #获得文件大小
                step = binfile.read(4)
                step = int().from_bytes(step, byteorder='little', signed=True)
                board = np.zeros((step, BOARD_SIZE * BOARD_SIZE))
                for i in range(step):
                    for j in range(BOARD_SIZE * BOARD_SIZE):
                        data = binfile.read(4)
                        data = int().from_bytes(data, byteorder='little', signed=True)
                        board[i][j] = data
                board = np.reshape(board,(-1,BOARD_SIZE,BOARD_SIZE))
                prob = np.zeros((step, BOARD_SIZE * BOARD_SIZE))
                for i in range(step):
                    for j in range(BOARD_SIZE * BOARD_SIZE):
                        data = binfile.read(4)
                        data = struct.unpack('f', data)[0]
                        prob[i][j] = data
                        # p = p.reshape((-1,BOARD_SIZE,BOARD_SIZE))
                    # print(p)

                v = []
                for i in range(step):
                    data = binfile.read(4)
                    data = int().from_bytes(data, byteorder='little', signed=True)
                    v.append(data)
                    # print(v)

                color = []
                for i in range(step):
                    data = binfile.read(4)
                    data = int().from_bytes(data, byteorder='little', signed=True)
                    color.append(data)

                last_action = []
                for i in range(step):
                    data = binfile.read(4)
                    data = int().from_bytes(data, byteorder='little', signed=True)
                    last_action.append(data)

                for i in range(step):
                    sym = self.get_symmetries(board[i], prob[i], last_action[i])
                    for b, p, a in sym:
                        train_examples.append([b, a, color[i], p, v[i]])
        return train_examples


if __name__ == '__main__':
    model_dir = path.join("..","build","weights")
    le = Learner(config.config)
    if len(sys.argv) <= 1 or sys.argv[1] == "prepare":
        print("save 0-th model !!")
        le.nnet.save_model(path.join(model_dir,'0'))
        print("done !")
    else:
        assert sys.argv[1] == "train", sys.argv[1]
        with open(path.join("..","build","current_and_best_weight.txt"), 'r') as f:
            current_id, best_id =  f.readline().split(" ")
            current_id = int(current_id)
        le.learn(model_dir=model_dir, model_id=current_id)
        with open(path.join("..","build","current_and_best_weight.txt"), 'w') as f:
            f.write(str(int(current_id)+1) + " "+ str(best_id))
        