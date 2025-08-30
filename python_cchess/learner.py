from collections import deque
from os import path, mkdir
import os
import torch
import numpy as np
import common as config
# import pickle
# import concurrent.futures
import random, struct
# from functools import reduce

import sys

sys.path.append('../build_cchess')

from neural_network import NeuralNetWorkWrapper


class Learner():
    def __init__(self, config):
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

        self.rows=10
        self.cols=9
        self.action_size = self.rows * self.cols * 35
        self.input_channel = 2*7+3
        self.input_size = self.rows * self.cols * self.input_channel

        # neural network
        self.batch_size = config['batch_size']
        self.epochs = config['epochs'] # *8 for no symmetry
        self.nnet = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
                                         config['num_channels'], self.rows, self.cols, 
                                         self.action_size, self.input_channel)

    def learn(self, model_dir,model_id):
        # train the model by self play

        model_path = path.join(model_dir, str(model_id))
        # assert path.exists(model_path+'.pkl'),f"{model_path+'.pkl'} does not exist!!!"
        # print(f"loading {model_id}-th model")
        self.nnet.load_model(model_path)

        data_path = path.join('..', 'build_cchess', 'data')
        train_data = self.load_samples(data_path)
        random.shuffle(train_data)

        # train neural network
        epochs = self.epochs * (len(train_data) + self.batch_size - 1) // self.batch_size
        self.nnet.train(train_data, min(self.batch_size,len(train_data)), int(epochs))

        model_path = path.join(model_dir, str(model_id+1))
        self.nnet.save_model(model_path)
        torch.cuda.empty_cache()

    # for go and gomoku only
    def get_symmetries(self, board, pi, last_action):
        # mirror, rotational
        assert (len(pi) == self.action_size)  # 1 for pass

        pi_board = np.reshape(pi, (self.n, self.n))
        last_action_board = np.zeros((self.n, self.n))
        if(last_action != -1):
            last_action_board[last_action // self.n][last_action % self.n] = 1
        l = []

        for i in [0]: #range(1, 5): # try:不做任何旋转,只有左右对称
            for j in [False]: #[True, False]:
                
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
        # BOARD_SIZE = self.n
        train_examples = []
        data_files = os.listdir(folder)
        for file_name in data_files:
            file_path = path.join(folder, file_name)
            with open(file_path, 'rb') as binfile:
                # size = os.path.getsize(filepath) #获得文件大小
                step = binfile.read(4)
                step = int().from_bytes(step, byteorder='little', signed=True)
                board = np.zeros((step, self.rows * self.cols))
                for st in range(step):
                    for j in range(self.rows * self.cols):
                        data = binfile.read(4)
                        data = int().from_bytes(data, byteorder='little', signed=True)
                        board[st][j] = data
                board = board.reshape(-1,self.rows,self.cols)
                prob = np.zeros((step, self.action_size))
                for st in range(step):
                    for j in range(self.action_size):
                        data = binfile.read(4)
                        data = struct.unpack('f', data)[0]
                        prob[st][j] = data

                v = []
                for st in range(step):
                    data = binfile.read(4)
                    data = int().from_bytes(data, byteorder='little', signed=True)
                    v.append(data)
                    # print(v)

                color = []
                for st in range(step):
                    data = binfile.read(4)
                    data = int().from_bytes(data, byteorder='little', signed=True)
                    color.append(data)

                last_action = []
                for st in range(step):
                    data = binfile.read(4)
                    data = int().from_bytes(data, byteorder='little', signed=True)
                    last_action.append(data)

                # for st in range(step):
                #     sym = self.get_symmetries(board[st], prob[st], last_action[st])
                #     for b, p, a in sym:
                #         train_examples.append([b, a, color[st], p, v[st]])
                no_cap_ratio = []
                for st in range(step):
                    data = binfile.read(4)
                    data=struct.unpack('f', data)[0]
                    no_cap_ratio.append(data)
                # print("no_cap_ratio =",no_cap_ratio[1])

                board_new = -np.ones((step, self.input_size))
                for st in range(step):
                    for i in range(self.input_size):
                        data = binfile.read(4)
                        data=struct.unpack('f', data)[0]
                        board_new[st][i] = data
                board_new = board_new.reshape((-1,self.input_channel,self.rows,self.cols))
                # print("board_new = \n",board_new[0][13,:,:]) # 13=下面的兵
                for st in range(step):
                    train_examples.append([board_new[st], last_action[st], color[st], prob[st], v[st]])
        return train_examples


if __name__ == '__main__':
    model_dir = path.join("..","build_cchess","weights")
    le = Learner(config.config)
    if len(sys.argv) <= 1 or sys.argv[1] == "prepare":
        print("save 0-th model !!")
        le.nnet.save_model(path.join(model_dir,'0'))
        print("prepare model done!")
    else:
        assert sys.argv[1] == "train", sys.argv[1]
        with open(path.join("..","build_cchess","current_and_best_weight.txt"), 'r') as f:
            current_id, best_id =  f.readline().split(" ")
            current_id = int(current_id)
        le.learn(model_dir=model_dir, model_id=current_id)
        with open(path.join("..","build_cchess","current_and_best_weight.txt"), 'w') as f:
            f.write(str(int(current_id)+1) + " "+ str(best_id))
        