#include <map>
#include <ctime>
#include <chrono>
#include <sstream>
#include <csignal>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/framework/tensor_util.h"

#include "tensorflow_graph_tools.hpp"
#include "GraphManager.hpp"
#include "ARFramework.hpp"
#include "grid_tools.hpp"

std::function<void(void)> shutdown_callback;
void shutdown_handler(int p)
{
    shutdown_callback();
    std::cout << "shutting down... please wait\n";
}


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
    std::string number_of_dimensions = "numdimensions";
    std::string class_averages = "class_averages";
    std::string label_proto = "label_proto";
    std::string label_layer = "label_layer_placeholder";
    std::string fgsm_balance_factor_opt = "0.95";
    std::string num_threads_str = "4";
    std::string num_abstractions_str = "1";
    std::string output_dir = "adv_examples";

    std::vector<tensorflow::Flag> flag_list = {
        tensorflow::Flag("graph", &graph, "path to protobuf graph to be executed"),
        tensorflow::Flag("input_layer", &input_layer, "name of input layer"),
        tensorflow::Flag("output_layer", &output_layer, "name of output layer"),
        tensorflow::Flag("gradient_layer", &gradient_layer, "name of the gradient layer (optional - used for FGSM)"),
        tensorflow::Flag("granularity", &granularity, "use this option is all dimensions share a discrete range"),
        tensorflow::Flag("verification_radius", &verification_radius, "'radius' of hyperrectangle within which safety is to be verified"),
        tensorflow::Flag("initial_activation", &initial_activation, "initial tested activation"),
        tensorflow::Flag("number_of_dimensions", &number_of_dimensions, "number of dimensions to verify with provided radius"),
        tensorflow::Flag("root_dir", &root_dir, "root_dir"),
        tensorflow::Flag("class_averages", &class_averages, "the class averages of the training data (optional - used for FGSM)"),
        tensorflow::Flag("label_proto", &label_proto, "protocol buffer of label image corresponding with initial activation"),
        tensorflow::Flag("label_layer", &label_layer, "name of label layer"),
        tensorflow::Flag("fgsm_balance_factor", &fgsm_balance_factor_opt, "Balance factor for modified FGSM algorithm (ratio dimensions fgsm/random)"),
        tensorflow::Flag("num_threads", &num_threads_str, "number of threads to use"),
        tensorflow::Flag("num_abstractions", &num_abstractions_str, "number of points to use as abstractions"),
        tensorflow::Flag("output_dir", &output_dir, "location of adversarial example output")
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
    auto radius = radiusProvided ? std::atof(verification_radius.c_str()) : 0.5;
    auto num_threads = std::atoi(num_threads_str.c_str());
    auto num_abstractions = std::atoi(num_abstractions_str.c_str());
    auto fgsm_balance_factor = std::atof(fgsm_balance_factor_opt.c_str());

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
    auto init_act_point = graph_tool::tensorToPoint(
            init_act_tensor);


    auto verifySubsetOfDimensions = number_of_dimensions != "numdimensions";
    auto numDimensionsToVerify = verifySubsetOfDimensions ? atoi(number_of_dimensions.c_str()) : init_act_point.size();
    auto numberOfInputDimensions = init_act_tensor.dims();
    auto flattenedNumDims = 1ull;

    // --------------
    std::cout << "########## Inital Point ##########\n";
    std::cout << "Number of dimensions in input: " << init_act_point.size() << "\n";
    std::cout << "Number of threads: " << num_threads << "\n";
    std::cout << "Number of points per abstractions: " << num_abstractions << "\n";
    std::cout << "Channels: " << numberOfInputDimensions << "\n";
    // --------------

    // first dimension is the batch size
    std::vector<int64_t> batch_input_shape(numberOfInputDimensions);
    // this is the size of a single input to the model
    std::vector<int64_t> input_shape(numberOfInputDimensions - 1);
    for(auto i = 0u; i < numberOfInputDimensions; ++i)
    {
        if(i != 0)
            input_shape[i - 1] = init_act_tensor.dim_size(i);
        batch_input_shape[i] = init_act_tensor.dim_size(i);
        flattenedNumDims *= batch_input_shape[i];
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

    for(auto i = 0u; i < 20u; ++i)
    {
        auto tmp = 
            gm.feedThroughModel(
                    retFeedDict, 
                    &graph_tool::parseGraphOutToVector, 
                    {output_layer});
        if(!gm.ok())
        {
            LOG(ERROR) << "Error while feeding through model";
            exit(1);
        }
        unsigned tmp_class = 
            graph_tool::getClassOfClassificationVector(tmp);

        if(tmp_class != orig_class)
            std::cout << tmp_class << " " << orig_class << "\n";
    }

    std::cout << "Granularity: " << granularityVal << "\n";
    std::cout << "Original class: " << orig_class << "\n";
    std::cout << "Input shape: ";
    for(auto&& elem : batch_input_shape)
        std::cout << elem << " ";
    std::cout << "\n";
    std::cout << "Number of dimensions: " << numDimensionsToVerify << "\n";
    std::cout << "Verification radius: " << radius << "\n";
    std::cout << "Root Directory: " << root_dir << "\n";
    std::cout << "Output Directory: " << output_dir << "\n";
    std::cout << "\n";

    /* difference between discrete values of each dimension */
    grid::point granularity_parsed;
    granularity_parsed.reserve(flattenedNumDims);
    std::fill_n(
            std::back_inserter(granularity_parsed), 
            flattenedNumDims, 
            granularityVal);

    grid::dimension_selection_strategy_t dimension_selection_strategy = 
        grid::largestDimFirst;
    /*
    grid::dimension_selection_strategy_t dimension_selection_strategy = 
        grid::randomDimSelection;
    */

    grid::dimension_selection_strategy_t 
        intellifeature_selection_strategy = 
        grid::randomDimSelection;

    grid::region_abstraction_strategy_t abstraction_strategy = 
        grid::RandomPointRegionAbstraction(num_abstractions);
    /*
    grid::region_abstraction_strategy_t abstraction_strategy = 
        grid::centralPointRegionAbstraction;
    */

    auto hasAveragesProto = class_averages != "class_averages";
    auto hasLabelProto = label_proto != "label_proto";
    auto hasLabelLayer = label_layer != "label_layer_placeholder";
    auto canUseGradient = 
        hasGradientLayer 
        && hasAveragesProto 
        && hasLabelProto 
        && hasLabelLayer;

    if(canUseGradient)
    {
        std::cout << "Using IntellifeatureFGSM\n";
        std::string class_averages_path = 
            tensorflow::io::JoinPath(root_dir, class_averages);
        auto class_averages_proto_pair = 
            GraphManager::ReadBinaryTensorProto(class_averages_path);
        if(!class_averages_proto_pair.first)
        {
            LOG(ERROR) << "Unable to read class averages proto";
            exit(1);
        }
        auto averages = graph_tool::tensorToPoints(
                class_averages_proto_pair.second);
        intellifeature_selection_strategy = 
            grid::IntellifeatureDimSelection(
                averages,
                &grid::l2norm,
                orig_class);
        auto label_tensor_path = 
            tensorflow::io::JoinPath(root_dir, label_proto);
        auto label_tensor_pair =
            GraphManager::ReadBinaryTensorProto(label_tensor_path);
        if(!label_tensor_pair.first)
        {
            LOG(ERROR) << "Unable to read label proto";
            exit(1);
        }
        auto label_tensor = label_tensor_pair.second;
        abstraction_strategy = 
            grid::ModifiedFGSMWithFallbackRegionAbstraction(
                num_abstractions,
                [&,label_tensor_copy = label_tensor]
                (grid::point const& p) -> grid::point
                {
                    auto createGradientFeedDict = 
                    [&]() -> graph_tool::feed_dict_type_t
                    {
                        auto p_tensor = 
                        graph_tool::pointToTensor(p, batch_input_shape);
                        return {{input_layer, p_tensor},
                            {label_layer, label_tensor_copy}};
                    };
                    auto retVal = gm.feedThroughModel(
                            createGradientFeedDict,
                            graph_tool::parseGraphOutToVector,
                            {gradient_layer});
                    if(!gm.ok())
                        LOG(ERROR) << "Error with model";
                    return retVal;
                },
                intellifeature_selection_strategy,
                grid::RandomPointRegionAbstraction(2u),
                granularity_parsed,
                fgsm_balance_factor);
    }

    grid::region_refinement_strategy_t refinement_strategy = 
        grid::HierarchicalDimensionRefinementStrategy(
                dimension_selection_strategy,
                2u,
                5u);

    auto all_valid_discretization_strategy = 
        grid::AllValidDiscretizedPointsAbstraction(
                graph_tool::tensorToPoint(init_act_tensor),
                granularity_parsed);
    
    // only attempt discrete search if total
    // valid points in region is less than a threshold
    const auto discrete_search_attempt_threshold = 1000ull;
    auto discrete_search_attempt_threshold_func = 
        [&](grid::region const& r)
        {
            return all_valid_discretization_strategy
                .getNumberValidPoints(r) 
                < discrete_search_attempt_threshold;
        };

    auto isPointSafe = [&](grid::point const& p)
            {
                auto logits_out = gm.feedThroughModel(
                        std::bind(graph_tool::makeFeedDict, 
                            input_layer, p, batch_input_shape),
                        &graph_tool::parseGraphOutToVector,
                        {output_layer});
                if(!gm.ok())
                    LOG(ERROR) << "GM Error in isPointSafe";
                auto class_out = 
                    graph_tool::getClassOfClassificationVector(logits_out);
                if(class_out != orig_class)
                {
                    std::cout << "found unsafe point\n";
                }
                return class_out == orig_class;
            };

    if(!isPointSafe(init_act_point))
    {
        LOG(ERROR) << "Original activation and original class do not agree\n";
        exit(1);
    }

    auto verification_engine = 
        grid::DiscreteSearchVerificationEngine(
                discrete_search_attempt_threshold_func,
                all_valid_discretization_strategy,
                isPointSafe);

    // create the initial region from the initial activation
    // and the user provided radius
    grid::region orig_region(init_act_point.size());
    for(auto i = 0u; i < orig_region.size(); ++i)
    {
        orig_region[i].first = 
            init_act_point[i] - radius;
        orig_region[i].second = 
            init_act_point[i] + radius;
    }

    grid::region domain_range(orig_region.size());
    for(auto i = 0u; i < domain_range.size(); ++i)
    {
        domain_range[i].first = 0;
        domain_range[i].second = 1;
    }

    orig_region = grid::snapToDomainRange(orig_region, domain_range);

    ARFramework arframework(
            gm,
            domain_range,
            init_act_point,
            granularity_parsed,
            orig_region,
            isPointSafe,
            verification_engine,
            abstraction_strategy,
            refinement_strategy
            );
    shutdown_callback = [&](){ arframework.join(); };
    auto handle = signal(SIGINT, shutdown_handler);
    std::vector<std::thread> thread_pool;

    for(auto i = 0u; i < num_threads; ++i)
    {
        thread_pool.emplace_back([&](){ arframework.run(); });
    }

    for(auto&& t : thread_pool)
        t.join();

    std::cout << "All threads joined\n";

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto now_tm = std::localtime(&now_c); 
    const unsigned BUFFER_SIZE = 30u;
    char buffer[BUFFER_SIZE];
    strftime(buffer, BUFFER_SIZE, "%Y_%m_%d_%H_%M_%S", now_tm);
    auto timestamp = std::string(buffer);
    auto report_function = [&](grid::point const& adv_exp)
    {
        static unsigned index = 0;
        std::vector<float> tmp(adv_exp.begin(), adv_exp.end());
        std::vector<std::size_t> tmp_size(batch_input_shape.begin(),
                batch_input_shape.end());
        auto adv_exp_tensor_proto = tensorflow::tensor::CreateTensorProto(
                tmp,
                tmp_size);
        auto classification = graph_tool::getClassOfClassificationVector(
                gm.feedThroughModel(
                    std::bind(graph_tool::makeFeedDict, 
                        input_layer, adv_exp, batch_input_shape),
                    &graph_tool::parseGraphOutToVector,
                    {output_layer})
                );
        if(!gm.ok())
            LOG(ERROR) << "GM: error in report function";
        std::stringstream file_name;
        file_name << timestamp 
            << "_" << index << "_" << orig_class << "_" 
            << classification << ".pb";
        ++index;
        auto file_path = tensorflow::io::JoinPath(output_dir, 
                file_name.str());
        auto write_status = 
            WriteBinaryProto(
                    tensorflow::Env::Default(), 
                    file_path,
                    adv_exp_tensor_proto);
        if(!write_status.ok())
        {
            LOG(ERROR) << "Couldn't write file " << file_path;
        }
    };
    std::cout << "reporting...\n";

    arframework.report(report_function);

    std::cout << "done\n";
    return 0;
}

