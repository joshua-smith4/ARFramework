#include "GraphManager.hpp"
#include <stdexcept>
#include <iostream>

GraphManager::GraphManager(std::string const& graph_file_name)
    : session(std::move(tensorflow::NewSession(tensorflow::SessionOptions())))
{
    tensorflow::GraphDef graph_def;
    auto load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) 
    {
        //throw std::domain_error("Error loading graph from file " + graph_file_name);
        std::cout << "ERROR: loading graph from file\n";
    }
    auto session_create_status = session->Create(graph_def);
    if(!session_create_status.ok())
    {
        //throw std::domain_error("Error creating session from GraphDef");
        std::cout << "ERROR: creating session from GraphDef\n";
    }
}

tensorflow::Tensor
GraphManager::ReadBinaryTensorProto(std::string const& path)
{
    tensorflow::TensorProto proto;
    auto load_tensorproto_status = 
        ReadBinaryProto(tensorflow::Env::Default(), path, &proto);
    if(!load_tensorproto_status.ok())
    {
        std::string err_msg = 
            "Failed to load TensorProto from file " + path;
        LOG(ERROR) << err_msg;
        //throw std::domain_error(err_msg);
        return tensorflow::Tensor();
    }
    tensorflow::Tensor retVal;
    auto success = retVal.FromProto(proto);
    if(!success)
    {
        std::string err_msg = 
             "Failed to read TensorProto into tensor.";
        LOG(ERROR) << err_msg;
        //throw std::domain_error(err_msg);
        return tensorflow::Tensor();
    }
    return retVal;
}


