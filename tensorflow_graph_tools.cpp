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


tensorflow::Tensor graph_tool::pointToTensorConversion(
        grid::point const& p,
        std::vector<int64_t> const& shape) // TensorShape expects initializer list
{
    tensorflow::Tensor retVal(tensorflow::DT_FLOAT,
            tensorflow::TensorShape(shape));
    auto flattened = retVal.flat<float>();
    for(auto i = 0u; i < p.size(); ++i)
        flattened(i) = (float)p[i];
    return retVal;
}

grid::point graph_tool::tensorToPointConversion(
        tensorflow::Tensor const& t)
{
    grid::point retVal;
    auto flattened = t.flat<float>();
    for(auto i = 0u; i < flattened.size(); ++i)
        retVal.push_back(flattened(i));
    return retVal;
}

graph_tool::feed_dict_type_t graph_tool::makeFeedDict(
        std::string const& input_name, 
        grid::point const& p,
        std::vector<int64_t> const& shape)
{
    return {{input_name, graph_tool::pointToTensorConversion(p, shape)}};
}

grid::point graph_tool::parseGraphOutToVector(
        std::vector<tensorflow::Tensor> const& out)
{
    return graph_tool::tensorToPointConversion(out[0]);
}

