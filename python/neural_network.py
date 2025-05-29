# -*- coding: utf-8 -*-
import sys
import os
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


def conv3x3(in_channels, out_channels, stride=1):
    # 3x3 convolution
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    # Residual block
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = False
        if in_channels != out_channels or stride != 1:
            self.downsample = True
            self.downsample_conv = conv3x3(in_channels, out_channels, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample_conv(residual)
            residual = self.downsample_bn(residual)

        out += residual
        out = self.relu(out)
        return out


class NeuralNetWork(nn.Module):
    """Policy and Value Network
    """

    def __init__(self, num_layers, num_channels, n, action_size,input_channel_size):
        super(NeuralNetWork, self).__init__()

        # residual block
        res_list = [ResidualBlock(input_channel_size, num_channels)] + [ResidualBlock(num_channels, num_channels) for _ in range(num_layers - 1)]
        self.res_layers = nn.Sequential(*res_list)

        # policy head
        self.p_conv = nn.Conv2d(num_channels, 4, kernel_size=1, padding=0, bias=False)
        self.p_bn = nn.BatchNorm2d(num_features=4)
        self.relu = nn.ReLU(inplace=True)

        self.p_fc = nn.Linear(4 * n ** 2, action_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # value head
        self.v_conv = nn.Conv2d(num_channels, 2, kernel_size=1, padding=0, bias=False)
        self.v_bn = nn.BatchNorm2d(num_features=2)

        self.v_fc1 = nn.Linear(2 * n ** 2, 256)
        self.v_fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        # residual block
        out = self.res_layers(inputs)

        # policy head
        p = self.p_conv(out)
        p = self.p_bn(p)
        p = self.relu(p)

        p = self.p_fc(p.view(p.size(0), -1))
        p = self.log_softmax(p)

        # value head
        v = self.v_conv(out)
        v = self.v_bn(v)
        v = self.relu(v)

        v = self.v_fc1(v.view(v.size(0), -1))
        v = self.relu(v)
        v = self.v_fc2(v)
        v = self.tanh(v)

        return p, v


class AlphaLoss(nn.Module):
    """
    Custom loss as defined in the paper :
    (z - v) ** 2 --> MSE Loss
    (-pi * logp) --> Cross Entropy Loss
    z : self_play_winner
    v : winner
    pi : self_play_probas
    p : probas

    The loss is then averaged over the entire batch
    """

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, log_ps, vs, target_ps, target_vs):
        value_loss = torch.mean(torch.pow(vs - target_vs, 2))
        policy_loss = -torch.mean(torch.sum(target_ps * log_ps, 1))

        return value_loss + policy_loss


class NeuralNetWorkWrapper():
    """train and predict
    """

    def __init__(self, lr, l2, num_layers, num_channels, n, action_size, input_channel_size=3):
        """ init
        """
        self.lr = lr
        self.l2 = l2
        self.num_channels = num_channels
        self.n = n
        self.input_channel_size = input_channel_size

        self.is_cuda_available = torch.cuda.is_available()
        if(self.is_cuda_available):
            print("发现并使用GPU")
        else:
            print("使用CPU")

        self.neural_network = NeuralNetWork(num_layers, num_channels, n, action_size, input_channel_size)
        if self.is_cuda_available:
            self.neural_network.cuda()

        self.optim = Adam(self.neural_network.parameters(), lr=self.lr, weight_decay=self.l2)
        self.alpha_loss = AlphaLoss()

    def train(self, example_buffer, batch_size, epochs):
        """train neural network
        """
        for epo in range(1, epochs + 1):
            self.neural_network.train()

            # sample
            train_data = random.sample(example_buffer, batch_size)


            # extract train data
            board_batch, last_action_batch, cur_player_batch, p_batch, v_batch = list(zip(*train_data))

            state_batch = self._data_convert(board_batch, last_action_batch, cur_player_batch)
            p_batch = np.array(p_batch)
            v_batch = np.array(v_batch)
            p_batch = torch.Tensor(p_batch).cuda() if self.is_cuda_available else torch.Tensor(p_batch)
            v_batch = torch.Tensor(v_batch).unsqueeze(
                1).cuda() if self.is_cuda_available else torch.Tensor(v_batch).unsqueeze(1)

            # zero the parameter gradients
            self.optim.zero_grad()

            # forward + backward + optimize
            log_ps, vs = self.neural_network(state_batch)
            loss = self.alpha_loss(log_ps, vs, p_batch, v_batch)
            loss.backward()

            self.optim.step()

            # calculate entropy
            new_p, _ = self._infer(state_batch)

            entropy = -np.mean(
                np.sum(new_p * np.log(new_p + 1e-10), axis=1)
            )

            print("EPOCH: {}, LOSS: {}, ENTROPY: {}".format(epo, loss.item(), entropy))

    def infer(self, feature_batch):
        """predict p and v by raw input
           return numpy
        """
        board_batch, last_action_batch, cur_player_batch = list(zip(*feature_batch))
        states = self._data_convert(board_batch, last_action_batch, cur_player_batch)

        self.neural_network.eval()
        log_ps, vs = self.neural_network(states)

        return np.exp(log_ps.cpu().detach().numpy()), vs.cpu().detach().numpy()

    def _infer(self, state_batch):
        """predict p and v by state
           return numpy object
        """

        self.neural_network.eval()
        log_ps, vs = self.neural_network(state_batch)

        return np.exp(log_ps.cpu().detach().numpy()), vs.cpu().detach().numpy()

    def _data_convert(self, board_batch, last_action_batch, cur_player_batch):
        """convert data format
           return tensor
        """
        n = self.n

        board_batch = torch.Tensor(np.array(board_batch)).unsqueeze(1)
        state0 = (board_batch > 0).float()
        state1 = (board_batch < 0).float()

        state2 = torch.zeros((len(last_action_batch), 1, n, n)).float()

        for i in range(len(board_batch)):
            if cur_player_batch[i] == -1:
                temp = state0[i].clone()
                state0[i].copy_(state1[i])
                state1[i].copy_(temp)

            last_action = last_action_batch[i]
            if last_action != -1:
                x, y = last_action // self.n, last_action % self.n
                state2[i][0][x][y] = 1

        res =  torch.cat((state0, state1, state2), dim=1)
        # res = torch.cat((state0, state1), dim=1)
        return res.cuda() if self.is_cuda_available else res

    def set_learning_rate(self, lr):
        """set learning rate
        """

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def load_model(self, filepath):
        """load model from file
        """

        state = torch.load(filepath+'.pkl')
        self.neural_network.load_state_dict(state['network'])
        self.optim.load_state_dict(state['optim'])
        if self.is_cuda_available:
            self.neural_network.cuda()


    def save_model(self, filepath):
        """save model to file
        """
        state = {'network':self.neural_network.state_dict(), 'optim':self.optim.state_dict()}
        torch.save(state, filepath+'.pkl')


        # save torchscript or onnx
        self.neural_network.eval()

        # if self.is_cuda_available:
        #     self.neural_network.cuda()
        #     example = torch.rand(1, self.input_channel_size, self.n, self.n).cuda()
        # else:

        self.neural_network.cpu()
        example = torch.rand(1, self.input_channel_size, self.n, self.n).cpu()
        dynamic_axes={"board":{0:"batch_size"},     # 批处理变量
                                    "P":{0:"batch_size"},"V":{0:"batch_size"}}
        torch.onnx.export(self.neural_network,
                          example,
                          filepath+'.onnx',
                          input_names=["board"],
                          output_names=['P', 'V'],
                          dynamic_axes=dynamic_axes)


if __name__ == '__main__':
    net=NeuralNetWorkWrapper(lr=0.1, l2=0.1, num_layers=3, num_channels=32, n=15, action_size=15*15)
    # print("save model")
    # net.save_model("/data/AlphaZero-Onnx/python/mymodel")

    print("load model")
    net.load_model("/data/AlphaZero-Onnx/python/mymodel")
    batch_all = 5
    state_batch = np.zeros((batch_all+1,3,15,15))

    state_batch[batch_all][1][0][0] = 1 # gomoku.execute_move(0);
    state_batch[batch_all][0][0][1] = 1 #   gomoku.execute_move(1);
    state_batch[batch_all][1][3][4] = 1 #   gomoku.execute_move(3*15+4=49);

    state_batch[batch_all][2][3][4] = 1 # last move 


    if net.is_cuda_available:
        state_batch = torch.Tensor(state_batch).cuda()
    else:
        state_batch = torch.Tensor(state_batch)
    print("predict")
    P,V = net._infer(state_batch) 
    print(f"P[{batch_all}:5]={P[batch_all][0:5]},V={V[batch_all][0]}")
    
