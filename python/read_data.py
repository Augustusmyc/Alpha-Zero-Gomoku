import struct
import os
import numpy as np

import struct



BOARD_SIZE = 15
if __name__ == '__main__':
    filepath = os.path.join('..', 'build', 'data',"data_0")
    step_id = 1
    with open(filepath, 'rb') as binfile:# read bin file
        size = os.path.getsize(filepath) # get the size of the bin file
        step = binfile.read(4)
        step = int().from_bytes(step, byteorder='little', signed=True)
        print("step = ",step)
        board = np.zeros((step,BOARD_SIZE*BOARD_SIZE))
        for i in range(step):
           for j in range(BOARD_SIZE*BOARD_SIZE):
               data = binfile.read(4)
               data = int().from_bytes(data, byteorder='little', signed=True)
               board[i][j] = data
        board = board.reshape((-1,BOARD_SIZE,BOARD_SIZE))
        print("board = ",board[step_id])

        p = np.zeros((step,BOARD_SIZE*BOARD_SIZE))
        for i in range(step):
            for j in range(BOARD_SIZE*BOARD_SIZE):
                data=binfile.read(4)  # TODO double 8?
                data=struct.unpack('f', data)[0]
                p[i][j] = data
        p = p.reshape((-1,BOARD_SIZE,BOARD_SIZE))
        print("p=", p[step_id])
        
        v = []
        for i in range(step):
            data = binfile.read(4)
            data = int().from_bytes(data, byteorder='little', signed=True)
            v.append(data)
        print("v=",v[step_id])

        color= []
        for i in range(step):
            data = binfile.read(4)
            data = int().from_bytes(data, byteorder='little', signed=True)
            color.append(data)
        print("color=",color[step_id])

        last_action = []

        for i in range(step):
            data = binfile.read(4)
            data = int().from_bytes(data, byteorder='little', signed=True)
            last_action.append(data)
        print("last_action=",last_action[step_id])