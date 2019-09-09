#ifndef GRAPH_MANAGER_HPP_INCLUDED
#define GRAPH_MANAGER_HPP_INCLUDED

#include <fstream>
#include <utility>
#include <vector>
#include <functional>
#include <memory>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

class GraphManager
{
public:
    GraphManager(std::string const&);
    static std::pair<bool, tensorflow::Tensor> ReadBinaryTensorProto(std::string const&);
    template <class InConvFunc, class OutConvFunc, class... In>
    typename 
    std::result_of<OutConvFunc(std::vector<tensorflow::Tensor>)>::type   
    feedThroughModel(
            InConvFunc&& in_func, 
            OutConvFunc&& out_func, 
            std::vector<std::string> const& output_labels,
            In&&... in_args) const
    {
        auto feed_dict = in_func(std::forward<In>(in_args)...);
        std::vector<tensorflow::Tensor> outputs;
        auto run_status = session->Run(
                feed_dict,
                output_labels,
                {},
                &outputs);
        if(!run_status.ok()) 
        {
            outputs.clear();
            errorOccurred = true;
        }
        return out_func(outputs);
    }
    inline bool ok() { return !errorOccurred; }
    inline void resetErrorFlag() { errorOccurred = false; }
private:
    std::unique_ptr<tensorflow::Session> session;
    bool errorOccurred;
};

#endif

