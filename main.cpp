// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <vector>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <codecvt>
using namespace std;
// #ifdef HAVE_TENSORRT_PROVIDER_FACTORY_H
// #include <tensorrt_provider_factory.h>
// #include <tensorrt_provider_options.h>

// std::unique_ptr<OrtTensorRTProviderOptionsV2> get_default_trt_provider_options() {
//   auto tensorrt_options = std::make_unique<OrtTensorRTProviderOptionsV2>();
//   tensorrt_options->device_id = 0;
//   tensorrt_options->has_user_compute_stream = 0;
//   tensorrt_options->user_compute_stream = nullptr;
//   tensorrt_options->trt_max_partition_iterations = 1000;
//   tensorrt_options->trt_min_subgraph_size = 1;
//   tensorrt_options->trt_max_workspace_size = 1 << 30;
//   tensorrt_options->trt_fp16_enable = false;
//   tensorrt_options->trt_int8_enable = false;
//   tensorrt_options->trt_int8_calibration_table_name = "";
//   tensorrt_options->trt_int8_use_native_calibration_table = false;
//   tensorrt_options->trt_dla_enable = false;
//   tensorrt_options->trt_dla_core = 0;
//   tensorrt_options->trt_dump_subgraphs = false;
//   tensorrt_options->trt_engine_cache_enable = false;
//   tensorrt_options->trt_engine_cache_path = "";
//   tensorrt_options->trt_engine_decryption_enable = false;
//   tensorrt_options->trt_engine_decryption_lib_path = "";
//   tensorrt_options->trt_force_sequential_engine_build = false;

//   return tensorrt_options;
// }
// #endif

void run_ort_trt() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  // const auto& api = Ort::GetApi();
  // OrtTensorRTProviderOptionsV2* tensorrt_options;
  
  
  // Ort::SessionOptions session_options
  Ort::SessionOptions *session_options = new Ort::SessionOptions();
  //session_options->SetIntraOpNumThreads(4);

  session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  
  
#ifdef _WIN32
  string model_path_s = "E:/Projects/AlphaZero-Onnx/python/mymodel.onnx";
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  //auto s = MultiByteToWideChar(model_path_s);
  const wchar_t* model_path = converter.from_bytes(model_path_s).c_str();
  auto sh = std::make_shared<Ort::Session>(Ort::Session(env, model_path, *session_options));
#else
  string model_path_s = "/data/AlphaZero-Onnx/python/mymodel.onnx";
  const char* model_path = model_path_s.c_str();
  auto cudaRet = OrtSessionOptionsAppendExecutionProvider_CUDA(*session_options, 0);
  cout<<"cuda id = " << cudaRet << endl;
  auto sh = std::make_shared<Ort::Session>(Ort::Session(env, model_path, *session_options));
#endif

    

// #ifdef _WIN32

//   const wchar_t* model_path = 
//   L"E:/Projects/AlphaZero-Onnx/python/mymodel.onnx";
// #else
//   const char* model_path = "/data/AlphaZero-Onnx/python/mymodel.onnx";
// #endif

  //*****************************************************************************************
  // It's not suggested to directly new OrtTensorRTProviderOptionsV2 to get provider options
  //*****************************************************************************************
  //
  // auto tensorrt_options = get_default_trt_provider_options();
  // session_options.AppendExecutionProvider_TensorRT_V2(*tensorrt_options.get());

  //**************************************************************************************************************************
  // It's suggested to use CreateTensorRTProviderOptions() to get provider options
  // since ORT takes care of valid options for you 
  //**************************************************************************************************************************
  
  // api.CreateTensorRTProviderOptions(&tensorrt_options);
  // std::unique_ptr<OrtTensorRTProviderOptionsV2, decltype(api.ReleaseTensorRTProviderOptions)> rel_trt_options(tensorrt_options, api.ReleaseTensorRTProviderOptions);
  // api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(session_options), rel_trt_options.get());

  // printf("Runing ORT TRT EP with default provider options\n");

  // Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)





  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = sh->GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 2, 15, 15}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = sh->GetInputName(i, allocator);
    printf("Input %d : name = %s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = sh->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type = %d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims = %zu\n", i, input_node_dims.size());
    for (size_t j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %zu = %jd\n", i, j, input_node_dims[j]);
  }

  size_t input_tensor_size = 3 * 15 * 15;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> output_node_names = {"V","P"};

  // initialize input data with values in [0.0, 1.0]
  for (unsigned int i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  input_node_dims[0] = 1;
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  // score model & input tensor, get back output tensor
  auto output_tensors = sh->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
  assert(output_tensors.size() == 2 && output_tensors[1].IsTensor());

  // Get pointer to output tensor float values
  float V = output_tensors[0].GetTensorMutableData<float>()[0];
  float* P = output_tensors[1].GetTensorMutableData<float>();
  //assert(abs(floatarr[0] - 0.000045) < 1e-6);

  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
     printf("P for board [%d] =  %f\n", i, P[i]);

  printf("V for board =  %f\n", V);

  // Results should be as below...
  // Score for class[0] = 0.000045
  // Score for class[1] = 0.003846
  // Score for class[2] = 0.000125
  // Score for class[3] = 0.001180
  // Score for class[4] = 0.001317


  // release buffers allocated by ORT alloctor
  for(const char* node_name : input_node_names)
    allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));

  printf("Done!\n");
}


int main(int argc, char* argv[]) {
  run_ort_trt();
  //cout<< "hello"<<endl;
  return 0;
}
