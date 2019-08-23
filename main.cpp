#include <iostream>
#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "GraphManager.hpp"

// command
//
/*

bazel-bin/tensorflow/ARFramework/gm_label_image --root_dir="/Users/JoshuaSmith/tensorflow/tensorflow/ARFramework" --graph="mnist_model.pb" --input_layer="input_mnist_input" --output_layer="k2tfout_0" --initial_activation="mnist_1000.pb"

*/

int main(int argc, char* argv[])
{
    std::string graph = "graph.pb";
    std::string input_layer = "input_layer";
    std::string output_layer = "output_layer";
    std::string initial_activation = "initial_activation.pb";
    std::string root_dir = ".";
    std::vector<tensorflow::Flag> flag_list = {
        tensorflow::Flag("graph", &graph, "path to protobuf graph to be executed"),
        tensorflow::Flag("input_layer", &input_layer, "name of input layer"),
        tensorflow::Flag("output_layer", &output_layer, "name of output layer"),
        tensorflow::Flag("initial_activation", &initial_activation, "initial tested activation"),
        tensorflow::Flag("root_dir", &root_dir, "root_dir"),
    };
    std::string usage = tensorflow::Flags::Usage(argv[0], flag_list);

    const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
    if (!parse_result) 
    {
        LOG(ERROR) << usage;
        return -1;
    }

    tensorflow::port::InitMain(argv[0], &argc, &argv);
    if (argc > 1) {
        LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
        return -1;
    }

    std::string graph_path = tensorflow::io::JoinPath(root_dir, graph);
    GraphManager gm(graph_path);
    if(!gm.ok())
    {
        LOG(ERROR) << "Error during construction";
        exit(1);
    }
    std::string initial_activation_path = 
        tensorflow::io::JoinPath(root_dir, initial_activation);
    auto init_act_tensor_status_pair = 
        GraphManager::ReadBinaryTensorProto(initial_activation_path);
    if(!init_act_tensor_status_pair.first)
    {
        LOG(ERROR) 
            << "Could not read initial activation protobuf: " 
            << initial_activation_path;
        exit(1);
    }
    auto init_act_tensor = init_act_tensor_status_pair.second;
    auto retFeedDict = [&]()
        ->std::vector<std::pair<std::string, tensorflow::Tensor>>
    {
        return {{input_layer, init_act_tensor}};
    };
    auto getRetVal = [](std::vector<tensorflow::Tensor> const& outs)
    {
        std::vector<std::vector<float>> ret;
        for(auto&& out : outs)
        {
            std::vector<float> tmp;
            auto flat = out.flat<float>();
            for(auto i = 0u; i < flat.size(); ++i)
                tmp.push_back(flat(i));
            ret.push_back(tmp);
        }
        return ret;
    };

    auto results = gm.feedThroughModel(retFeedDict, getRetVal, {output_layer});
    if(!gm.ok())
    {
        LOG(ERROR) << "Error while feeding through model";
        exit(1);
    }
    for(auto&& res : results)
    {
        for(auto&& elem : res)
            std::cout << elem << " ";
        std::cout << "\n";
    }
    return 0;
}

