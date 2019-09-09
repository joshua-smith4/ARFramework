#ifndef TENSORFLOW_GRAPH_TOOLS_INCLUDED
#define TENSORFLOW_GRAPH_TOOLS_INCLUDED

#include "tensorflow/core/framework/tensor.pb.h"
#include "grid_tools.hpp"

namespace graph_tool
{
    unsigned getClassOfClassificationTensor(
            tensorflow::Tensor const&);

    template <class T>
    unsigned getClassOfClassificationVector(
            std::vector<T> const& classes)
    {
        auto max_elem = std::max_element(classes.begin(), classes.end());
        return std::distance(classes.begin(), max_elem);
    }

    tensorflow::Tensor pointToTensorConversion(
            grid::point const&,
            std::vector<int> const&);

    grid::point tensorToPointConversion(
            tensorflow::Tensor const&);
}

#endif

