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

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

std::pair<Status, Tensor> ReadBinaryTensorProto(std::string const& path)
{
    tensorflow::TensorProto proto;
    Status load_tensorproto_status = 
        ReadBinaryProto(tensorflow::Env::Default(), path, &proto);
    if(!load_tensorproto_status.ok())
        return {load_tensorproto_status, Tensor()};
    Tensor retVal;
    auto success = retVal.FromProto(proto);
    if(!success)
    {
        LOG(ERROR) << "failed from proto";
        return { Status::OK(), Tensor() };
    }
    return { Status::OK(), retVal };
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
std::pair<Status, std::unique_ptr<tensorflow::Session>> LoadGraph(const string& graph_file_name)
{
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return {tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'"), nullptr};
  }
  std::unique_ptr<tensorflow::Session> session(std::move(tensorflow::NewSession(tensorflow::SessionOptions())));
  Status session_create_status = session->Create(graph_def);
  return {session_create_status, std::move(session)};
}

int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
  string graph =
      "output_graph.pb";
  string input_layer_x = "x_input";
  string input_layer_y = "y_input";
  string output_layer_z = "z_out";
  string output_layer_oz = "oz_out";
  string root_dir = ".";
  std::vector<Flag> flag_list = {
      Flag("graph", &graph, "graph to be executed"),
      Flag("input_layer_x", &input_layer_x, "name of input layer_x"),
      Flag("input_layer_y", &input_layer_y, "name of input layer_y"),
      Flag("output_layer_z", &output_layer_z, "name of output layer_z"),
      Flag("output_layer_oz", &output_layer_oz, "name of output layer_oz"),
      Flag("root_dir", &root_dir, "root_dir"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // First we load and initialize the model.
  string graph_path = tensorflow::io::JoinPath(root_dir, graph);
  auto status_session = LoadGraph(graph_path);
  if (!status_session.first.ok()) {
    LOG(ERROR) << status_session.first;
    return -1;
  }
  auto x = ReadBinaryTensorProto("x.pb");
  auto y = ReadBinaryTensorProto("y.pb");
  if(!(x.first.ok() && y.first.ok()))
  {
      LOG(ERROR) << "Reading didn't work";
  }
  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  Status run_status = status_session.second->Run(
          {
            {input_layer_x, x.second}, 
            {input_layer_y, y.second},
          },
          {output_layer_z, output_layer_oz},
          {}, 
          &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }
  LOG(INFO) << "printing outputs";

  for(auto&& output : outputs)
  {
      auto float_output = output.flat<float>();
      LOG(INFO) << float_output(0);
  }
  return 0;
}
