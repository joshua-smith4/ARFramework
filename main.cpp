#include <map>
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

#include "tensorflow_graph_tools.hpp"
#include "GraphManager.hpp"
#include "grid_tools.hpp"

std::set<grid::point, grid::region_less_compare> 
    foundAdversarialExamples;

void interrupt_handler(int p)
{
    for(auto&& adv_example : foundAdversarialExamples)
    {
        std::cout << "Found adv example\n";
    }
    exit(1);
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
        tensorflow::Flag("fgsm_balance_factor", &fgsm_balance_factor_opt, "Balance factor for modified FGSM algorithm (ratio dimensions fgsm/random)")
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

    // tmp stuff
    // --------------
    std::cout << "########## Inital Point ##########\n";
    std::cout << init_act_point.size() << "\n";
    std::cout << numberOfInputDimensions << "\n";
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

    /*
    auto back_check = graph_tool::pointToTensor(init_act_point,
            batch_input_shape);

    auto btmp = back_check.flat<float>();
    auto tmp = init_act_tensor.flat<float>();
    for(auto i = 0u; i < tmp.size(); ++i)
    {
        std::cout << btmp(i) << " " << tmp(i) << "\n";
    }
    */

    auto retFeedDict = [&]()
        ->std::vector<std::pair<std::string, tensorflow::Tensor>>
    {
        return {{input_layer, init_act_tensor}};
    };

    std::cout << "about to feed in init activation\n";
    auto logits_init_activation = 
        gm.feedThroughModel(
                retFeedDict, 
                &graph_tool::parseGraphOutToVector, 
                {output_layer});
    std::cout << "done\n";
    if(!gm.ok())
    {
        LOG(ERROR) << "Error while feeding through model";
        exit(1);
    }

    unsigned orig_class = 
        graph_tool::getClassOfClassificationVector(logits_init_activation);

    for(auto i = 0u; i < 10u; ++i)
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

    std::cout << "granularity: " << granularityVal << "\n";
    std::cout << "original class: " << orig_class << "\n";
    std::cout << "input shape: ";
    for(auto&& elem : batch_input_shape)
        std::cout << elem << " ";
    std::cout << "\n";
    std::cout << "number_of_dimensions: " << numDimensionsToVerify << "\n";
    std::cout << "verification_radius: " << radius << "\n\n";

    /* difference between discrete values of each dimension */
    grid::point granularity_parsed;
    granularity_parsed.reserve(flattenedNumDims);
    std::fill_n(
            std::back_inserter(granularity_parsed), 
            flattenedNumDims, 
            granularityVal);

    grid::dimension_selection_strategy_t dimension_selection_strategy = 
        grid::randomDimSelection;

    grid::dimension_selection_strategy_t 
        intellifeature_selection_strategy = 
        grid::randomDimSelection;

    grid::region_abstraction_strategy_t abstraction_strategy = 
        grid::RandomPointRegionAbstraction(20);

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
            grid::ModifiedFGSMRegionAbstraction(
                5u,
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
                    return gm.feedThroughModel(
                            createGradientFeedDict,
                            graph_tool::parseGraphOutToVector,
                            {gradient_layer});
                },
                intellifeature_selection_strategy,
                fgsm_balance_factor);
    }

    grid::region_refinement_strategy_t refinement_strategy = 
        grid::HierarchicalDimensionRefinementStrategy(
                dimension_selection_strategy,
                2,
                8);

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
                auto class_out = 
                    graph_tool::getClassOfClassificationVector(logits_out);
                std::cout << "class of initial activation: " << class_out << "\n";
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

    std::vector<grid::region> potentiallyUnsafeRegions;
    std::vector<grid::region> safeRegions;
    unsigned long long numberSafeValidPoints = 0ull;
    std::vector<std::pair<grid::region, grid::point>> 
        unsafeRegionsWithAdvExamples;

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
    auto selected_dimensions = 
        intellifeature_selection_strategy(
                orig_region, 
                init_act_point.size());

    for(auto i = numDimensionsToVerify; i < selected_dimensions.size(); ++i)
    {
        orig_region[selected_dimensions[i]].first = 
            init_act_point[selected_dimensions[i]];
        orig_region[selected_dimensions[i]].second = 
            init_act_point[selected_dimensions[i]];
    }
    std::cout << "\n";

    grid::region domain_range(orig_region.size());
    for(auto i = 0u; i < domain_range.size(); ++i)
    {
        domain_range[i].first = 0;
        domain_range[i].second = 1;
    }

    orig_region = grid::snapToDomainRange(orig_region, domain_range);
    if(!grid::isValidRegion(orig_region))
    {
        LOG(ERROR) << "the generated verification region is invalid";
        exit(1);
    }
    std::cout << "Total number of valid points in verification region: "
        << grid::AllValidDiscretizedPointsAbstraction
        ::getNumberValidPoints(orig_region, init_act_point, granularity_parsed)
        << "\n";

    potentiallyUnsafeRegions.push_back(orig_region);

    std::cout << "about to start main process\n";
    auto handle = signal(SIGINT, interrupt_handler);
    const auto PRINT_PERIOD = 100ull;
    auto print_count = PRINT_PERIOD;
    while(!potentiallyUnsafeRegions.empty() || 
            !unsafeRegionsWithAdvExamples.empty())
    {
        if(print_count >= PRINT_PERIOD)
        {
            std::cout << "Number of found adversarial examples: " 
                << foundAdversarialExamples.size() << "\n";
            std::cout << "Number of potentially unsafe regions: " 
                << potentiallyUnsafeRegions.size() << "\n";
            std::cout << "Number of unsafe regions: " 
                << unsafeRegionsWithAdvExamples.size() << "\n";
            std::cout << "Number of safe points: " 
                << numberSafeValidPoints << "\n";
            std::cout << "Number of safe regions: " 
                << safeRegions.size() << "\n";
            print_count = 0ull;
        }
        print_count++;
        if(!potentiallyUnsafeRegions.empty())
        {
            auto selected_region = potentiallyUnsafeRegions.back();
            potentiallyUnsafeRegions.pop_back();

            selected_region = grid::snapToDomainRange(
                    selected_region,
                    domain_range);

            auto numValidPoints = grid
                ::AllValidDiscretizedPointsAbstraction
                ::getNumberValidPoints(
                        selected_region,
                        init_act_point,
                        granularity_parsed);
            if(numValidPoints <= 0) continue;
            auto verification_result = 
                verification_engine(selected_region);
            if(verification_result.first == 
                    grid::VERIFICATION_RETURN::SAFE)
            {
                safeRegions.push_back(selected_region);
                numberSafeValidPoints += numValidPoints;
            }
            else if(verification_result.first == 
                    grid::VERIFICATION_RETURN::UNSAFE)
            {
                foundAdversarialExamples.insert(
                        verification_result.second);    
                auto subregions = refinement_strategy(selected_region);
                auto subregion_with_adv_exp = 
                    subregions.find(verification_result.second);
                if(subregions.end() == subregion_with_adv_exp)
                {
                    LOG(ERROR) << "Adversarial example was found that did not belong to any subregions";
                }
                else
                {
                    unsafeRegionsWithAdvExamples.push_back({*subregion_with_adv_exp, verification_result.second});
                    subregions.erase(subregion_with_adv_exp);
                }
                std::copy(subregions.begin(), subregions.end(),
                        std::back_inserter(potentiallyUnsafeRegions));
            }
            else if(verification_result.first == 
                    grid::VERIFICATION_RETURN::UNKNOWN)
            {
                auto subregions = refinement_strategy(selected_region);
                std::map<grid::region, grid::point> unsafeRegionsTmp;
                std::vector<grid::point> all_abstracted_points;
                for(auto&& subregion : subregions)
                {
                    auto abstracted_points = 
                        abstraction_strategy(subregion);
                    //std::cout << "abstracted points: " << abstracted_points.size() << "\n";
                    for(auto&& pt : abstracted_points)
                    {
                        auto snapped_pt = grid::snapToDomainRange(
                                grid::enforceSnapDiscreteGrid(
                                    pt,
                                    init_act_point,
                                    granularity_parsed),
                                domain_range);
                        if(grid::isInDomainRange(snapped_pt, domain_range))
                        {
                            all_abstracted_points.push_back(snapped_pt);
                        }
                        else
                        {
                            LOG(ERROR) 
                                << "Point was outside domain range\n";
                        }
                    }
                }
                //std::cout << "about to enter all abstracted points\n";
                for(auto&& pt : all_abstracted_points)
                {
                    //print_point(pt);
                    auto found_subregion = subregions.find(pt);
                    if(found_subregion == subregions.end()) continue;
                    foundAdversarialExamples.insert(pt);
                    unsafeRegionsTmp.insert({*found_subregion, pt});
                    subregions.erase(found_subregion);
                }
                std::copy(unsafeRegionsTmp.begin(),
                        unsafeRegionsTmp.end(),
                        std::back_inserter(unsafeRegionsWithAdvExamples));
                std::copy(subregions.begin(), 
                        subregions.end(),
                        std::back_inserter(potentiallyUnsafeRegions));
            }
        }
        else
        {
            auto selected_region_adv_exp_pair = 
                unsafeRegionsWithAdvExamples.back();
            unsafeRegionsWithAdvExamples.pop_back();

            if(grid::AllValidDiscretizedPointsAbstraction
                    ::getNumberValidPoints(
                        selected_region_adv_exp_pair.first, 
                        init_act_point, 
                        granularity_parsed) 
                    <= 1ull)
            {
                //std::cout << "region has one or fewer valid points\n";
                continue;
            }
            auto subregions = refinement_strategy(
                    selected_region_adv_exp_pair.first);
            std::set<grid::region, grid::region_less_compare> 
                nonempty_subregions;
            for(auto&& subregion : subregions)
            {
                if(grid::AllValidDiscretizedPointsAbstraction
                        ::getNumberValidPoints(
                            subregion,
                            init_act_point,
                            granularity_parsed) != 0ull)
                    nonempty_subregions.insert(subregion);
            }

            if(nonempty_subregions.empty()) continue;
            auto unsafeRegionItr = nonempty_subregions.find(
                    selected_region_adv_exp_pair.second);
            if(unsafeRegionItr != nonempty_subregions.end())
            {
                unsafeRegionsWithAdvExamples.push_back(
                        {*unsafeRegionItr, 
                        selected_region_adv_exp_pair.second});
                nonempty_subregions.erase(unsafeRegionItr);
            }
            else
            {
                LOG(ERROR) << "Adversarial example was found not belonging to region after refined";
            }

            std::copy(subregions.begin(), subregions.end(),
                    std::back_inserter(potentiallyUnsafeRegions));
        }
    }

    std::cout << "Number of found adversarial examples: " 
        << foundAdversarialExamples.size() << "\n";
    std::cout << "Number of potentially unsafe regions: " 
        << potentiallyUnsafeRegions.size() << "\n";
    std::cout << "Number of unsafe regions: " 
        << unsafeRegionsWithAdvExamples.size() << "\n";
    std::cout << "Number of safe points: " 
        << numberSafeValidPoints << "\n";
    std::cout << "Number of safe regions: " 
        << safeRegions.size() << "\n";

    std::cout << "exiting\n";
    return 0;
}

