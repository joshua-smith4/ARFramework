#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/tensor.pb.h"

#include "GraphManager.hpp"

GraphManager::GraphManager(std::string const& graph_file_name)
    : session(std::move(tensorflow::NewSession(tensorflow::SessionOptions()))),
    errorOccurred(false)
{
    tensorflow::GraphDef graph_def;
    auto load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) 
    {
        LOG(FATAL) << "Could not load graph from file: " << graph_file_name;
        errorOccurred = true;
    }
    auto session_create_status = session->Create(graph_def);
    if(!session_create_status.ok())
    {
        LOG(FATAL) << "Could not create graph from GraphDef";
        errorOccurred = true;
    }
}

std::pair<bool, tensorflow::Tensor>
GraphManager::ReadBinaryTensorProto(std::string const& path)
{
    tensorflow::TensorProto proto;
    auto load_tensorproto_status = 
        ReadBinaryProto(tensorflow::Env::Default(), path, &proto);
    if(!load_tensorproto_status.ok())
    {
        LOG(ERROR) << "Failed to load TensorProto from file " + path;
        return {false, tensorflow::Tensor()};
    }
    tensorflow::Tensor retVal;
    auto success = retVal.FromProto(proto);
    if(!success)
    {
        LOG(ERROR) << "Failed to read TensorProto into tensor.";
        return {false, tensorflow::Tensor()};
    }
    return {true, retVal};
}


