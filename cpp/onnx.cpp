//#pragma once
#include <iostream>
#include <common.h>
#include <random>
#include <onnx.h>
#include <onnxruntime_cxx_api.h>
#include <future>
#include <memory>
#include <queue>
#include <assert.h>
#include <algorithm>

#include <codecvt>

// using namespace customType;

NeuralNetwork::NeuralNetwork(const std::string model_path, const unsigned int batch_size)
    : // module(std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path.c_str()))),
      env(nullptr),
      shared_session(nullptr),
      batch_size(batch_size),
      running(true),
      loop(nullptr),
      memory_info(nullptr)
{
  memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  this->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "alphaZero");
  // const auto& api = Ort::GetApi();
  // OrtTensorRTProviderOptionsV2* tensorrt_options;
  Ort::SessionOptions *session_options = new Ort::SessionOptions();
  // auto share_session_options = std::make_shared<Ort::SessionOptions>(session_options);
  //session_options->SetIntraOpNumThreads(1); // TODO:study the parameter
  //session_options->SetInterOpNumThreads(1); // TODO:study the parameter

  session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  
#ifdef _WIN32
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    const wchar_t* model_path_w = converter.from_bytes(model_path).c_str();
    //No CUDA
    shared_session = std::make_shared<Ort::Session>(Ort::Session(env, model_path_w, *session_options));
#else
    auto ret = OrtSessionOptionsAppendExecutionProvider_CUDA(*session_options, 0); // TODO: update the old API...
    std::cout << "CUDA id = " << ret <<std::endl;
    // Ort::Session session = Ort::Session(env, model_path.c_str(), *session_options);
    shared_session = std::make_shared<Ort::Session>(Ort::Session(env, model_path.c_str(), *session_options));
#endif
  //sess = &session;

  size_t input_tensor_size = CHANNEL_SIZE * BORAD_SIZE * BORAD_SIZE; 
  // simplify ... using known dim values to calculate size
  // use OrtGetTensorShapeElementCount() to get official size!

  // this->input_tensor_values = std::vector<float> (input_tensor_size);
  this->output_node_names = std::vector<const char *>{"V", "P"};

  // print number of model input nodes
  size_t num_input_nodes = shared_session->GetInputCount();
  this->input_node_names = std::vector<const char *>(num_input_nodes);
  // simplify... this model has only 1 input node {?, 3, 15, 15}.
  // Otherwise need vector<vector<>>

  //printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++)
  {
    // print input node names
    char *input_name = shared_session->GetInputName(i, allocator);
    // printf("Input %d : name = %s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = shared_session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    // printf("Input %d : type = %d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    // printf("Input %d : num_dims = %zu\n", i, input_node_dims.size());
    // for (size_t j = 0; j < input_node_dims.size(); j++)
    //   printf("Input %d : dim %zu = %jd\n", i, j, input_node_dims[j]);
  }

  /////////////

  // input_node_dims[0] = 1;

  // std::vector<float> input_tensor_values(input_tensor_size);

  // for (unsigned int i = 0; i < input_tensor_size; i++)
  //   input_tensor_values[i] = (float)i / (input_tensor_size + 1);
  // auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // input_node_dims[0] = 1;
  // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  // assert(input_tensor.IsTensor());

  // // score model & input tensor, get back output tensor
  // auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
  // assert(output_tensors.size() == 2 && output_tensors[1].IsTensor());
  // std::cout<<"ok!!"<<std::endl;
/////////////


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
}

std::future<NeuralNetwork::return_type> NeuralNetwork::commit(Gomoku *gomoku)
{
  std::vector<float> state = transorm_gomoku_to_Tensor(gomoku);

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

std::vector<float> NeuralNetwork::transorm_board_to_Tensor(board_type board, int last_move, int cur_player)
{
  auto input_tensor_values = std::vector<float>(CHANNEL_SIZE * BORAD_SIZE * BORAD_SIZE);
  int first = 0;
  int second = 0;
  if (cur_player == BLACK)
  {
    second = 1; //Black currently play = All black positions occupy the 0-th dimension in board
  }
  else
  {
    first = 1;
  }
  for (int r = 0; r < BORAD_SIZE; r++)
  {
    for (int c = 0; c < BORAD_SIZE; c++)
    {
      switch (board[r][c])
      {
      case 1:
        input_tensor_values[first * BORAD_SIZE * BORAD_SIZE + r * BORAD_SIZE + c] = 1;
        break;
      case -1:
        input_tensor_values[second * BORAD_SIZE * BORAD_SIZE + r * BORAD_SIZE + c] = 1;
        break;
      default:
        break;
      }
    }
    if(last_move >=0){
      input_tensor_values[2 * BORAD_SIZE * BORAD_SIZE + last_move] = 1;
    }
  }
  return input_tensor_values;

  // std::vector<int> board0;
  // std::vector<int> state0;
  // std::vector<int> state1;
  // for (unsigned int i = 0; i < BORAD_SIZE; i++) {
  //     board0.insert(board0.end(), board[i].begin(), board[i].end());
  // }

  // torch::Tensor temp =
  //     torch::from_blob(&board0[0], { 1, 1, BORAD_SIZE, BORAD_SIZE }, torch::dtype(torch::kInt32));

  // torch::Tensor state0 = temp.gt(0).toType(torch::kFloat32);
  // torch::Tensor state1 = temp.lt(0).toType(torch::kFloat32);

  // if (cur_player == -1) {
  //     std::swap(state0, state1);
  // }

  // torch::Tensor state2 =
  //     torch::zeros({ 1, 1, BORAD_SIZE, BORAD_SIZE }, torch::dtype(torch::kFloat32));

  // if (last_move != -1) {
  //     state2[0][0][last_move / BORAD_SIZE][last_move % BORAD_SIZE] = 1;
  // }

  // torch::Tensor states = torch::cat({ state0, state1 }, 1);
  //  return cat({ state0, state1, state2 }, 1);
}

std::vector<float> NeuralNetwork::transorm_gomoku_to_Tensor(Gomoku *gomoku)
{
  return NeuralNetwork::transorm_board_to_Tensor(gomoku->get_board(), gomoku->get_last_move(), gomoku->get_current_color());
}

void NeuralNetwork::infer()
{
  //{
  //  this->module->eval();
  // torch::NoGradGuard no_grad;
  // torch::AutoGradMode enable_grad(false);
  // get inputs
  // std::vector<torch::Tensor> states;
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



  size_t input_tensor_size = input_node_dims[0] * CHANNEL_SIZE * BORAD_SIZE * BORAD_SIZE;
  std::vector<float> state_all(0);
  for (auto &item : states)
  {
    state_all.insert(state_all.end(), item.begin(), item.end());
  }

  // std::for_each(state_all.begin(), state_all.end(), [](double x) { std::cout << x << ","; });
  // std::cout << std::endl;

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, state_all.data(), input_tensor_size, this->input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  auto output_tensors = shared_session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
  
  assert(output_tensors.size() == 2 && output_tensors[0].IsTensor() && output_tensors[1].IsTensor());

  float *V = output_tensors[0].GetTensorMutableData<float>();
  float *P = output_tensors[1].GetTensorMutableData<float>();

  for (unsigned int i = 0; i < promises.size(); i++)
  {

    // Get pointer to output tensor float values

    std::vector<double> prob(P + i * BORAD_SIZE * BORAD_SIZE, P + (i+1)*BORAD_SIZE * BORAD_SIZE);
    std::vector<double> value{V[i]};
    //assert(prob.size() == BORAD_SIZE * BORAD_SIZE);

    for (int j = 0; j < BORAD_SIZE * BORAD_SIZE; j++)
    {
      prob[j] = std::exp(prob[j]);
      // printf("prob [%d] =  %f\n", j, prob[j]);
    }
    //printf("value [%d] =  %f\n", 0, V[0]);
    // printf("value [%d] =  %f\n", 1, V[1]);

    // printf("prob [%d] =  %f\n", 0, P[0]);
    // printf("prob [%d] =  %f\n", 1, P[1]);
    // printf("prob [%d] =  %f\n", 2, P[2]);
    // printf("prob [%d] =  %f\n", 3, P[3]);

    return_type temp{std::move(prob), std::move(value)};

    promises[i].set_value(std::move(temp));
  }

  // #ifdef USE_GPU
  //     TS inputs{ cat(states, 0).to(at::kCUDA) };
  // #else
  //   TS inputs{ cat(states, 0) };
  // #endif

  // #ifdef JIT_MODE
  //     auto result = this->module->forward(inputs).toTuple();
  //     torch::Tensor p_batch = result->elements()[0]
  //         .toTensor()
  //         .exp()
  //         .toType(torch::kFloat32)
  //         .to(at::kCPU);
  //     torch::Tensor v_batch =
  //         result->elements()[1].toTensor().toType(torch::kFloat32).to(at::kCPU);
  // #else
  //     auto result = this->module->forward(inputs);
  //     //std::cout << y.requires_grad() << std::endl; // prints `false`

  //     Tensor p_batch = result.first.exp().toType(kFloat32).to(at::kCPU);
  //     Tensor v_batch = result.second.toType(kFloat32).to(at::kCPU);
  // #endif

  //   // set promise value
  //   for (unsigned int i = 0; i < promises.size(); i++) {
  //     torch::Tensor p = p_batch[i];
  //     torch::Tensor v = v_batch[i];

  //     std::vector<double> prob(static_cast<float*>(p.data_ptr()),
  //                              static_cast<float*>(p.data_ptr()) + p.size(0));
  //     std::vector<double> value{v.item<float>()};

  //     return_type temp{std::move(prob), std::move(value)};

  //     promises[i].set_value(std::move(temp));
  //   }
}
