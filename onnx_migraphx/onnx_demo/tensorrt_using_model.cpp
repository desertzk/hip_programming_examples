#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <memory>

//g++ -std=c++14 tensorrt_using_model.cpp -o tensorrt_add_demo -I/usr/src/tensorrt/include -I/usr/local/cuda/include -L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda/lib64 -lnvinfer -lnvonnxparser -lcudart

// A simple logger class for TensorRT
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // Suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

// Helper function to read a file into a buffer
bool readFile(const std::string& filename, std::vector<char>& buffer)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
    {
        std::cerr << "ERROR: Could not open file " << filename << std::endl;
        return false;
    }
    auto size = file.tellg();
    buffer.resize(size);
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), size);
    file.close();
    return true;
}

int main()
{
    Logger logger;

    // 1) Create builder, network, and parser
    // These are raw pointers, and we are responsible for deleting them.
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    if (!builder)
    {
        std::cerr << "ERROR: could not create builder." << std::endl;
        return -1;
    }

    // Create the network. kEXPLICIT_BATCH is deprecated and now the default behavior.
    auto network = builder->createNetworkV2(0U);
    if (!network)
    {
        std::cerr << "ERROR: could not create network." << std::endl;
        return -1;
    }

    auto parser = nvonnxparser::createParser(*network, logger);
    if (!parser)
    {
        std::cerr << "ERROR: could not create parser." << std::endl;
        return -1;
    }

    // Parse the ONNX model file
    if (!parser->parseFromFile("add.onnx", static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr << "ERROR: could not parse ONNX model." << std::endl;
        return -1;
    }

    // 2) Configure builder
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    if (!config)
    {
        std::cerr << "ERROR: could not create builder config." << std::endl;
        return -1;
    }
    // Set the memory pool limit. Replaces deprecated setMaxWorkspaceSize.
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 20); // 1 MB

    // 3) Build engine and execution context
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine)
    {
        std::cerr << "ERROR: could not build engine." << std::endl;
        return -1;
    }

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (!context)
    {
        std::cerr << "ERROR: could not create execution context." << std::endl;
        return -1;
    }

    // 4) Prepare input data
    const int rows = 2;
    const int cols = 3;
    const int dataSize = rows * cols;

    std::vector<float> inputX(dataSize, 1.0f);
    std::vector<float> inputY(dataSize, 2.0f);
    std::vector<float> output(dataSize);

    // 5) Allocate device buffers
    void* buffers[3];
    // Use getTensorIndex to get binding indices from the engine
    int idxX = 0, idxY = 1, idxZ = 2;



    cudaMalloc(&buffers[idxX], dataSize * sizeof(float));
    cudaMalloc(&buffers[idxY], dataSize * sizeof(float));
    cudaMalloc(&buffers[idxZ], dataSize * sizeof(float));

    // 6) Copy inputs to device
    cudaMemcpy(buffers[idxX], inputX.data(), dataSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(buffers[idxY], inputY.data(), dataSize * sizeof(float), cudaMemcpyHostToDevice);

    // 7) Run inference
    context->executeV2(buffers);

    // 8) Retrieve output from device
    cudaMemcpy(output.data(), buffers[idxZ], dataSize * sizeof(float), cudaMemcpyDeviceToHost);

    // 9) Print output and verify
    std::cout << "Output:";
    float max_error = 0.0f;
    for (int i = 0; i < dataSize; ++i)
    {
        std::cout << ' ' << output[i];
        max_error = std::max(max_error, std::abs(output[i] - (inputX[i] + inputY[i])));
    }
    std::cout << std::endl;
    std::cout << "Max absolute error: " << max_error << std::endl;


    // 10) Cleanup: Use delete for objects created with create* functions
    cudaFree(buffers[idxX]);
    cudaFree(buffers[idxY]);
    cudaFree(buffers[idxZ]);
    
    delete context;
    delete engine;
    delete config;
    delete parser;
    delete network;
    delete builder;

    return 0;
}
