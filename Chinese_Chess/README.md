# Alpha Zero Chinese Chess
Chinese Chess AI (Alpha Zero) implemented by pytorch and onnxruntime.

# Supported OS System
linux/Windows (tested on Ubuntu 22 + GPU and Windows + GPU or CPU)


# Dependence
gcc (linux) or visual studio 19+ (windows)

cmake 4.0.0+

pytorch (tested on onnxruntime-v1.22.0)

onnxruntime-gpu (tested on onnxruntime-v1.22.0)


# Installation
Download and install miniconda / python
and "pip install" all the dependent packages such as pytorch，onnx ...

train.sh: convert "python" to "/data/miniconda3/bin/python" or "python3" or your own python intepreter path

Download onnxruntime: https://github.com/microsoft/onnxruntime/releases/tag/v1.22.0

convert onnxruntime path in CMakefiles.txt to your own path：
set(ONNXRUNTIME_ROOTDIR "/mnt/data/myc/projects/onnxruntime-linux-x64-gpu-1.22.0")


mkdir build

cp *.sh ./build

cd ./build

cmake ..    (or  win10: "cmake -A x64 ..")

cmake --build . --config Release   (or win10: open .sln file through visual Studio 19 and generate)


# Train (Linux)
cd ./build

chmod 777 train.sh

bash train.sh

If you want to train the model on windows 10/11, convert "train.sh" to "train.bat" and transfer commands.


# Human play with AI (inference)
run mcts_test, for example in linux:

./mcts_test ./weights/423.onnx 1 

Here 1(or 0) = AI play with black(or white) pieces. 

The newest trained model will be updated in "model" directory. 

Increase or decrease "NUM_MCT_SIMS" in include/common.h (default 1600) to increase the power or speed of AI.

