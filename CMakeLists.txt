cmake_minimum_required(VERSION 4.0.0)
project(AlphaZeroGomoku)

set(CMAKE_BUILD_TYPE Release)




if(CMAKE_COMPILER_IS_GNUCC)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -pthread")
    set(ONNXRUNTIME_ROOTDIR "/data/myc/onnxruntime-linux-x64-gpu-1.22.0")
else()
    set(ONNXRUNTIME_ROOTDIR "E:/Projects/onnxruntime-win-x64-1.22.0")
endif()



# tensorrt_provider_factory.h contains old APIs of the tensorrt execution provider
# include(CheckIncludeFileCXX)
# CHECK_INCLUDE_FILE_CXX(tensorrt_provider_factory.h HAVE_TENSORRT_PROVIDER_FACTORY_H)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories("${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")


# add sources
include_directories(./include)
aux_source_directory(./cpp SOURCES)

add_library(test_lib ${SOURCES})
ADD_EXECUTABLE(train_eval_net  ./test/train_eval_net.cpp)
ADD_EXECUTABLE(mcts_test  ./test/mcts_test.cpp)
# ADD_EXECUTABLE(thread_pool_test  ./test/thread_pool_test.cpp)
# ADD_EXECUTABLE(gomoku_test  ./test/gomoku_test.cpp)

target_link_libraries(train_eval_net test_lib)
target_link_libraries(mcts_test test_lib)
# target_link_libraries(thread_pool_test test_lib)
# target_link_libraries(gomoku_test test_lib)

target_link_libraries(train_eval_net onnxruntime)
target_link_libraries(mcts_test onnxruntime)

if (MSVC)
 file(GLOB ONNX_DLLS "${ONNXRUNTIME_ROOTDIR}/lib/*.dll")
 add_custom_command(TARGET test_lib POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${ONNX_DLLS} ${CMAKE_CURRENT_BINARY_DIR}/Release)
endif (MSVC)


# ADD_EXECUTABLE(AlphaZeroInference  main.cpp)

# target_link_libraries(AlphaZeroInference onnxruntime)


# if(HAVE_TENSORRT_PROVIDER_FACTORY_H)
#   target_compile_definitions(AlphaZeroInference PRIVATE -DHAVE_TENSORRT_PROVIDER_FACTORY_H)
# endif()
