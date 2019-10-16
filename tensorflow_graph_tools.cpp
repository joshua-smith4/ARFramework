#include "tensorflow_graph_tools.hpp"

unsigned graph_tool::getClassOfClassificationTensor(
        tensorflow::Tensor const& t)
{
    unsigned retVal = 0u;
    float maxVal = 0;
    auto flattened = t.flat<float>();
    maxVal = flattened(0);
    for(auto i = 0u; i < flattened.size(); ++i)
    {
        if(flattened(i) > maxVal)
        {
            maxVal = flattened(i);
            retVal = i;
        }
    }
    return retVal;
}


tensorflow::Tensor graph_tool::pointToTensor(
        grid::point const& p,
        std::vector<int64_t> const& shape) 
{
    tensorflow::Tensor retVal(tensorflow::DT_FLOAT,
            tensorflow::TensorShape(shape));
    auto flattened = retVal.flat<float>();
    for(auto i = 0u; i < p.size(); ++i)
        flattened(i) = (float)p[i];
    return retVal;
}

tensorflow::Tensor graph_tool::pointsToTensor(
        std::vector<grid::point> const& p,
        std::vector<int64_t> const& shapeOfEachPoint) 
{
    std::vector<int64_t> shape = {static_cast<int64_t>(p.size())};
    std::copy(shapeOfEachPoint.begin(), shapeOfEachPoint.end(),
            std::back_inserter(shape));
    tensorflow::Tensor retVal(tensorflow::DT_FLOAT,
            tensorflow::TensorShape(shape));
    auto flattened = retVal.flat<float>();
    for(auto i = 0u; i < p.size(); ++i)
        for(auto j = 0u; j < p[i].size(); ++j)
            flattened(i) = (float)p[i][j];
    return retVal;
}

grid::point graph_tool::tensorToPoint(
        tensorflow::Tensor const& t)
{
    grid::point retVal;
    auto flattened = t.flat<float>();
    for(auto i = 0u; i < flattened.size(); ++i)
        retVal.push_back(flattened(i));
    return retVal;
}

std::vector<grid::point> graph_tool::tensorToPoints(
        tensorflow::Tensor const& t)
{
    std::vector<grid::point> retVal;
    auto numPoints = t.dim_size(0);
    auto pointLength = 1u;
    for(auto i = 1u; i < t.dims(); ++i)
        pointLength *= t.dim_size(i);
    auto flattened = t.flat<float>();
    for(auto i = 0u; i < numPoints; ++i)
    {
        grid::point tmp(pointLength);
        for(auto j = 0u; j < pointLength; ++j)
            tmp[j] = flattened(i*pointLength + j);
        retVal.push_back(tmp);
    }
    return retVal;
}

graph_tool::feed_dict_type_t graph_tool::makeFeedDict(
        std::string const& input_name, 
        grid::point const& p,
        std::vector<int64_t> const& shape)
{
    return {{input_name, graph_tool::pointToTensor(p, shape)}};
}

grid::point graph_tool::parseGraphOutToVector(
        std::vector<tensorflow::Tensor> const& out)
{
    return graph_tool::tensorToPoint(out[0]);
}

