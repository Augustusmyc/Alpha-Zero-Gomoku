import struct
import os
import numpy as np

import struct



rows = 10
cols = 9
mode_num = 35

input_channel = 2*7+3
input_size = rows * cols * input_channel
if __name__ == '__main__':
    filepath = "/mnt/data/myc/projects/Alpha-Zero-Gomoku/build_cchess/data/data_0"
    step_id = 1
    with open(filepath, 'rb') as binfile:# read bin file
        size = os.path.getsize(filepath) # get the size of the bin file
        step = binfile.read(4)
        step = int().from_bytes(step, byteorder='little', signed=True)
        print("step = ",step)
        board = 9*np.ones((step,rows*cols))
        for st in range(step):
           for i in range(rows*cols):
               data = binfile.read(4)
               data = int().from_bytes(data, byteorder='little', signed=True)
               board[st][i] = data
        board = board.reshape((-1,rows,cols))
        print("board = \n",board[step_id])

        p = 9*np.ones((step,rows*cols*mode_num))
        for st in range(step):
            for i in range(rows*cols*mode_num):
                data=binfile.read(4)  # TODO double 8?
                data=struct.unpack('f', data)[0]
                p[st][i] = data
        p = p.reshape((-1,rows,cols,mode_num))

        np.set_printoptions(linewidth=np.inf)  # 设置行宽为无限，避免换行(avoid change line)
        print("p =\n", np.max(p[step_id],axis=-1))
        
        v = []
        for st in range(step):
            data = binfile.read(4)
            data = int().from_bytes(data, byteorder='little', signed=True)
            v.append(data)
        print("v=",v[step_id])

        color= []
        for st in range(step):
            data = binfile.read(4)
            data = int().from_bytes(data, byteorder='little', signed=True)
            color.append(data)
        print("color =",color[step_id])

        last_action = []

        for st in range(step):
            data = binfile.read(4)
            data = int().from_bytes(data, byteorder='little', signed=True)
            last_action.append(data)
        print("last_action =",last_action[step_id])

        no_cap_ratio = []
        for st in range(step):
            data = binfile.read(4)
            data=struct.unpack('f', data)[0]
            no_cap_ratio.append(data)
        print("no_cap_ratio =",no_cap_ratio[step_id])

        board_new = -np.ones((step, input_size))
        for st in range(step):
           for i in range(input_size):
               data = binfile.read(4)
               data=struct.unpack('f', data)[0]
               board_new[st][i] = data
        board_new = board_new.reshape((-1,input_channel,rows,cols))
        # print(board_new.shape)
        print("board_new = \n",board_new[step_id][13,:,:]) # 13=下面的兵