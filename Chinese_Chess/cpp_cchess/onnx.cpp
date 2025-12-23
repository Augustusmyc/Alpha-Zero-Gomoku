//#pragma once
#include <iostream>
// #include <common.h>
#include <random>
#include <onnx.h>
#include <onnxruntime_cxx_api.h>
#include <future>
#include <memory>
#include <queue>
#include <assert.h>
#include <algorithm>
#include <codecvt>
#include "chinese_chess.h"

NeuralNetwork::NeuralNetwork(const std::string model_path, const unsigned int batch_size)
    : env(nullptr),
      shared_session(nullptr),
      batch_size(batch_size),
      running(true),
      loop(nullptr),
      memory_info(nullptr)
{
  memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  this->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "alphaZero");
  Ort::SessionOptions *session_options = new Ort::SessionOptions();
  session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  
#ifdef _WIN32
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    // const wchar_t* model_path_w = converter.from_bytes(model_path).c_str(); AI建议更改
    const char* model_path_cstr = model_path.c_str();
    //No CUDA
    shared_session = std::make_shared<Ort::Session>(Ort::Session(env, model_path_w, *session_options));
#else
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    session_options->AppendExecutionProvider_CUDA(cuda_options);
    session_options->DisableCpuMemArena();
    session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    shared_session = std::make_shared<Ort::Session>(env, model_path.c_str(), *session_options);
#endif
  size_t input_tensor_size = ChineseChess::input_size;
  // simplify ... using known dim values to calculate size
  // use OrtGetTensorShapeElementCount() to get official size!

  // this->input_tensor_values = std::vector<float> (input_tensor_size);
  this->output_node_names = std::vector<const char *>{"V", "P"};

  // print number of model input nodes
  size_t num_input_nodes = shared_session->GetInputCount();
  this->input_node_names = std::vector<const char *>(num_input_nodes);
  // simplify... this model has only 1 input node {?, characters, rows, cols}.
  // Otherwise need vector<vector<>>

  //printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++)
  {
    // print input node names
    auto allocated_name = shared_session->GetInputNameAllocated(i, allocator);
    input_node_names[i] = strdup(allocated_name.get()); // 需要持久化存储

    // print input node types
    Ort::TypeInfo type_info = shared_session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    // printf("Input %d : num_dims = %zu\n", i, input_node_dims.size());
    // for (size_t j = 0; j < input_node_dims.size(); j++)
    //   printf("Input %d : dim %zu = %jd\n", i, j, input_node_dims[j]);
  }


  // run infer thread
  this->loop = std::make_unique<std::thread>([this]
                                             {
    while (this->running) {
      this->infer();
    } });
}

NeuralNetwork::~NeuralNetwork()
{
  this->running = false;
  this->loop->join();
  for(const char* node_name : input_node_names) {
        free(const_cast<char*>(node_name)); 
    }
}

std::future<NeuralNetwork::return_type> NeuralNetwork::commit(ChineseChess *game)
{
  std::vector<float> state = game->transorm_game_to_Tensor();

  // emplace task
  std::promise<return_type> promise;
  auto ret = promise.get_future();

  {
    std::lock_guard<std::mutex> lock(this->lock);
    tasks.emplace(std::make_pair(std::move(state), std::move(promise)));
  }

  this->cv.notify_all();
  // std::cout<< "return promise !" << std::endl;
  return ret;
}

void NeuralNetwork::infer()
{
  std::vector<std::vector<float>> states; // TODO class
  std::vector<std::promise<return_type>> promises;
  bool timeout = false;

  while (states.size() < this->batch_size && !timeout)
  {
    // pop task
    {
      std::unique_lock<std::mutex> lock(this->lock);
      if (this->cv.wait_for(lock, std::chrono::microseconds(1),
                            [this]
                            { return this->tasks.size() > 0; }))
      {
        auto task = std::move(this->tasks.front());
        states.emplace_back(std::move(task.first));
        promises.emplace_back(std::move(task.second));

        this->tasks.pop();
      }
      else
      {
        // timeout
        // std::cout << "timeout" << std::endl;
        timeout = true;
      }
    }
  }
  // inputs empty
  size_t size = states.size();
  if (size == 0)
  {
    return;
  }


  // set promise value
  this->input_node_dims[0] = promises.size();

  //std::cout<<"promises size  = "<<promises.size()<<std::endl;



  size_t input_tensor_size = input_node_dims[0] * ChineseChess::input_size;
  std::vector<float> state_all(0);
  for (auto &item : states)
  {
    state_all.insert(state_all.end(), item.begin(), item.end());
  }

  // std::for_each(state_all.begin(), state_all.end(), [](double x) { std::cout << x << ","; });
  // std::cout << std::endl;

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, state_all.data(), input_tensor_size, this->input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  // std::vector<Ort::Value> output_tensors = shared_session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
  std::vector<Ort::Value> output_tensors;
  // output_tensors = shared_session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
  try {
    output_tensors = shared_session->Run(Ort::RunOptions{nullptr}, 
                                input_node_names.data(), 
                                &input_tensor, 
                                1, 
                                output_node_names.data(), 
                                2);
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error !! : " << e.what() << std::endl;
        throw;
    }
  
  assert(output_tensors.size() == 2 && output_tensors[0].IsTensor() && output_tensors[1].IsTensor());

  float *V = output_tensors[0].GetTensorMutableData<float>();
  float *P = output_tensors[1].GetTensorMutableData<float>();

  // promises.size() == batch
  for (unsigned int i = 0; i < promises.size(); i++)
  {

    // Get pointer to output tensor float values

    std::vector<double> prob(P + i * ChineseChess::action_size, P + (i+1)*ChineseChess::action_size);
    std::vector<double> value{V[i]};
    //assert(prob.size() == BORAD_SIZE * BORAD_SIZE); //for gomoku

    for (int j = 0; j < ChineseChess::action_size; j++)
    {
      prob[j] = std::exp(prob[j]);
      // printf("prob [%d] =  %f\n", j, prob[j]);
    }
    //printf("value [%d] =  %f\n", 0, V[0]);
    // printf("value [%d] =  %f\n", 1, V[1]);

    // printf("prob [%d] =  %f\n", 0, P[0]);
    // printf("prob [%d] =  %f\n", 1, P[1]);

    return_type temp{std::move(prob), std::move(value)};

    promises[i].set_value(std::move(temp));
  }
}