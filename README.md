# Alpha-Zero-Gomoku
Gomoku AI (Alpha Zero) implemented by pytorch and onnxruntime.

# Tutorial About Alpha Zero:
bilibili (part 1-3):

https://www.bilibili.com/video/BV1a5411Q7en/
https://www.bilibili.com/video/BV1tL4y1K7UB/
https://www.bilibili.com/video/BV1Fa411L7ea/

# Supported Games
Currently only Gomoku and similar games such as Tic-Tac-Toe. 

Welcome other game implementions if you want to become the contributor!


# Supported OS System
linux/Windows (tested on Ubuntu 20 + GPU and Windows 10 + GPU or CPU)


# Supported Enviroment
Both GPU and CPU (GPU test on Tesla V100 + Cuda 11 / CPU test on Intel i7)


# Language
C++ (for speed!) and python. The model is trained by pytorch (Python) and onnxruntime (C++,for selfplay), and inferenced by onnxruntime (C++).


# Dependence
gcc(linux) or visual studio 19(windows)

cmake 3.13+

pytorch (tested on 1.11 and 1.7)

onnxruntime-gpu (tested on 1.11)


# Installation
change the onnxruntime path in CMakefiles.txt

mkdir build

copy *.sh to ./build

cd ./build

cmake ..    (or "cmake -A x64 ..")

cmake --build . --config Release   (or open .sln file through visual Studio 19 and generate for win10)


# Train (Linux)
cd ./build

bash train_net.sh

If you want to train the model on windows 10, transform "train.sh" to "train.bat" and fix corresponding commands.


# Human play with AI (inference)
run mcts_test, for example in linux:

./mcts_test ./weights/1000.onnx 1

Here 1(or 0) = AI play with black(or white) pieces. 

The newest trained model are put in "model" directory. 

Increase or decrease "NUM_MCT_SIMS" in include/common.h (default 1600) to increase the power or speed of AI.

# About This Project (Chinese)

一个使用pytorch + onnxruntime训练的Alpha Zero训练框架。
onnxruntime主要负责“左右互搏”的部分，pytorch负责模型参数优化。

目前游戏仅支持五子棋和井字棋，如果有其它小伙伴愿意提供其它棋类游戏的源码，这里非常欢迎。

支持多线程蒙特卡洛树搜索,该部分和模型推理（用于左右互搏）部分均由c++完成，之所以不用python主要是为了加快速度，并且避开python GIL的坑。

本项目支持windows和linux平台。由于我主要在linux上训练，windows相应的代码可能未来不会及时更新，需要修改一下才能用。

