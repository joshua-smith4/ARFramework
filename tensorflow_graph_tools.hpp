#ifndef TENSORFLOW_GRAPH_TOOLS_INCLUDED
#define TENSORFLOW_GRAPH_TOOLS_INCLUDED

#include <cstdint>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "grid_tools.hpp"

namespace graph_tool
{
    unsigned getClassOfClassificationTensor(tensorflow::Tensor const&);

    template <class T>
    unsigned getClassOfClassificationVector(
            std::vector<T> const& classes)
    {
        auto max_elem = std::max_element(classes.begin(), classes.end());
        return std::distance(classes.begin(), max_elem);
    }

    tensorflow::Tensor pointToTensor(
            grid::point const&,
            std::vector<int64_t> const&);

    tensorflow::Tensor pointsToTensor(
            std::vector<grid::point> const&,
            std::vector<int64_t> const&);

    grid::point tensorToPoint(
            tensorflow::Tensor const&);

    std::vector<grid::point> tensorToPoints(
            tensorflow::Tensor const&);
    
    using feed_dict_type_t = std::vector<std::pair<std::string, tensorflow::Tensor>>;
    
    feed_dict_type_t makeFeedDict(
            std::string const&, 
            grid::point const&, 
            std::vector<int64_t> const&);

    grid::point parseGraphOutToVector(std::vector<tensorflow::Tensor> const&);
    
}

#endif

