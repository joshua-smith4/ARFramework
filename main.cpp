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

int main(int argc, char* argv[])
{
    std::string graph = "output_graph.pb";
    std::string input_layer_x = "x_input";
    std::string input_layer_y = "y_input";
    std::string output_layer_z = "z_out";
    std::string output_layer_oz = "oz_out";
    std::string root_dir = ".";
    std::vector<tensorflow::Flag> flag_list = {
        tensorflow::Flag("graph", &graph, "graph to be executed"),
        tensorflow::Flag("input_layer_x", &input_layer_x, "name of input layer_x"),
        tensorflow::Flag("input_layer_y", &input_layer_y, "name of input layer_y"),
        tensorflow::Flag("output_layer_z", &output_layer_z, "name of output layer_z"),
        tensorflow::Flag("output_layer_oz", &output_layer_oz, "name of output layer_oz"),
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
    std::string x_path = tensorflow::io::JoinPath(root_dir, "x.pb");
    std::string y_path = tensorflow::io::JoinPath(root_dir, "y.pb");
    auto x = GraphManager::ReadBinaryTensorProto(x_path);
    auto y = GraphManager::ReadBinaryTensorProto(y_path);
    auto retFeedDict = [&]()
        ->std::vector<std::pair<std::string, tensorflow::Tensor>>
    {
        return {{input_layer_x, x}, {input_layer_y, y}};
    };
    auto getRetVal = [](std::vector<tensorflow::Tensor> const& outs)
    {
        std::vector<float> ret;
        for(auto&& out : outs)
        {
            ret.push_back(out.flat<float>()(0));
        }
        return ret;
    };

    auto results = gm.feedThroughModel(retFeedDict, getRetVal, {output_layer_z, output_layer_oz});
    for(auto&& res : results)
    {
        std::cout << res << " ";
    }
    std::cout << "\n";
    return 0;
}



