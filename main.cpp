#include <iostream>
#include <fstream>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow_graph_tools.hpp"
#include "GraphManager.hpp"
#include "grid_tools.hpp"

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
    std::string gradient_layer = "gradient_layer_tmp_placeholder";
    std::string granularity = "granularity";
    std::string verification_radius = "verification radius";
    std::string initial_activation = "initial_activation.pb";
    std::string root_dir = ".";

    std::vector<tensorflow::Flag> flag_list = {
        tensorflow::Flag("graph", &graph, "path to protobuf graph to be executed"),
        tensorflow::Flag("input_layer", &input_layer, "name of input layer"),
        tensorflow::Flag("output_layer", &output_layer, "name of output layer"),
        tensorflow::Flag("gradient_layer", &gradient_layer, "name of the gradient layer (optional - used for FGSM)"),
        tensorflow::Flag("granularity", &granularity, "use this option is all dimensions share a discrete range"),
        tensorflow::Flag("verification_radius", &verification_radius, "'radius' of hyperrectangle within which safety is to be verified"),
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

    auto hasGradientLayer = gradient_layer != "gradient_layer_tmp_placeholder";
    auto granularityProvided = granularity != "granularity";
    double granularityVal = granularityProvided ? atof(granularity.c_str()) : 1.0;

    auto radiusProvided = verification_radius != "verification radius";
    if(!radiusProvided)
    {
        LOG(ERROR) << "Must provide verification radius.";
        return -1;
    }
    auto radius = radiusProvided ? atof(verification_radius.c_str()) : 0.5;

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
    auto numberOfInputDimensions = init_act_tensor.dims();
    auto flattenedNumDims = 1ull;
    std::vector<long long> input_shape(numberOfInputDimensions);
    for(auto i = 0u; i < numberOfInputDimensions; ++i)
    {
        input_shape[i] = init_act_tensor.dim_size(i);
        flattenedNumDims *= input_shape[i];
    }

    auto retFeedDict = [&]()
        ->std::vector<std::pair<std::string, tensorflow::Tensor>>
    {
        return {{input_layer, init_act_tensor}};
    };

    auto logits_init_activation = 
        gm.feedThroughModel(
                retFeedDict, 
                &graph_tool::parseGraphOutToVector, 
                {output_layer});
    if(!gm.ok())
    {
        LOG(ERROR) << "Error while feeding through model";
        exit(1);
    }

    unsigned orig_class = 
        graph_tool::getClassOfClassificationVector(logits_init_activation);


    std::cout << "granularity: " << granularityVal << "\n";
    std::cout << "original class: " << orig_class << "\n";
    std::cout << "input shape: ";
    for(auto&& elem : input_shape)
        std::cout << elem << " ";
    std::cout << "\n";
    std::vector<grid::point> foundAdversarialExamples;

    /* difference between discrete values of each dimension */
    grid::point granularity_parsed;
    granularity_parsed.reserve(flattenedNumDims);
    std::fill_n(std::back_inserter(granularity_parsed), flattenedNumDims, granularityVal);

    grid::dimension_selection_strategy_t dimension_selection_strategy = 
        grid::maxAverageDimSelection;
    if(hasGradientLayer)
        dimension_selection_strategy = 
            grid::IntellifeatureDimSelection(
                {}, // TODO: fill with class averages
                &grid::l2norm,
                orig_class);

    grid::region_abstraction_strategy_t abstraction_strategy = 
        grid::centralPointRegionAbstraction;
    if(hasGradientLayer)
        abstraction_strategy = 
            grid::ModifiedFGSMRegionAbstraction(
                20u,
                [&](grid::point const& p) -> grid::point
                {
                    return gm.feedThroughModel(
                            std::bind(graph_tool::makeFeedDict, input_layer, p, input_shape),
                            graph_tool::parseGraphOutToVector,
                            {gradient_layer});
                },
                dimension_selection_strategy,
                0.8);

    grid::region_refinement_strategy_t refinement_strategy = 
        grid::HierarchicalDimensionRefinementStrategy(
                dimension_selection_strategy,
                2,
                4);

    auto all_valid_discretization_strategy = 
        grid::AllValidDiscretizedPointsAbstraction(
                graph_tool::tensorToPointConversion(init_act_tensor),
                granularity_parsed);
    
    // only attempt discrete search if total
    // valid points in region is less than a threshold
    const auto discrete_search_attempt_threshold = 100000ull;
    auto discrete_search_attempt_threshold_func = 
        [&](grid::region const& r)
        {
            return all_valid_discretization_strategy
                .getNumberValidPoints(r) 
                < discrete_search_attempt_threshold;
        };

    auto verification_engine = 
        grid::DiscreteSearchVerificationEngine(
                discrete_search_attempt_threshold_func,
                all_valid_discretization_strategy,
                [&](grid::point const& p)
                {
                    auto logits_out = gm.feedThroughModel(
                            std::bind(graph_tool::makeFeedDict, input_layer, p, input_shape),
                            &graph_tool::parseGraphOutToVector,
                            {output_layer});
                    auto class_out = 
                        graph_tool::getClassOfClassificationVector(logits_out);
                    return class_out == orig_class;
                });

    return 0;
}

