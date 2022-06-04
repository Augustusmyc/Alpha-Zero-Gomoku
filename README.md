# Alpha-Zero-Gomoku
Gomoku AI (Alpha Zero) implemented by pytorch and onnxruntime.

# Supported Games
Currently only Gomoku and similar games such as Tic-Tac-Toe. 

Welcome other game implementions if you want to become the contributor!


# Supported OS System
linux/Windows (tested on Ubuntu 20 + GPU and Windows 10 + GPU or CPU)


# Supported Enviroment
Both GPU and CPU (GPU test on Tesla V100 + Cuda 11 / CPU test on Intel i7)


# Language
C++ (for speed!) and python. The model is trained by pytorch(Python) and onnxruntime(C++,for selfplay), and inferenced by onnxruntime(C++).


# Dependence
gcc(linux) or visual studio 19(windows)

cmake 3.13+

pytorch (tested on 1.11 and 1.7)


# Installation
mkdir build

copy *.sh to ./build

cd ./build

cmake ..    (or "cmake -A x64 ..")

cmake --build . --config Release   (or open .sln file through visual Studio 19 and generate for win10)


# Train (Linux)
cd ./build

bash train_net.sh

If you want to train the model on windows 10, transform "train.sh" to "train.bat" and fix corresponding commands.


# Human play with AI
run mcts_test, for example in linux:

./mcts_test ./weights/1000.onnx 1

Here 1(or 0) = AI play with black(or white) pieces. 