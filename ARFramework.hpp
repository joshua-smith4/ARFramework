#ifndef ARFRAMWORK_HPP_INCLUDED
#define ARFRAMWORK_HPP_INCLUDED

#include <map>
#include <thread>
#include <atomic>
#include <mutex>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/command_line_flags.h"

#include "tensorflow_graph_tools.hpp"
#include "GraphManager.hpp"
#include "grid_tools.hpp"

class ARFramework
{
private:
    std::set<grid::region, grid::region_less_compare> 
        potentiallyUnsafeRegions;
    std::mutex pur_mutex;
    std::vector<grid::region> safeRegions;
    std::mutex sr_mutex;
    std::map<grid::region, grid::point, grid::region_less_compare> 
        unsafeRegionsWithAdvExamples;
    std::mutex ur_mutex;
    bool keep_working;
    GraphManager& gm;
    grid::region domain_range;
    grid::point init_point;
    grid::point granularity;

    grid::region_abstraction_strategy_t abstraction_strategy;
    grid::region_refinement_strategy_t refinement_strategy;
    grid::verification_engine_type_t verification_engine;

    std::function<bool(grid::point const&)> safety_predicate;
    std::thread::id logging_thread_id;
    std::atomic_flag log_thread_set;

    void worker_routine();
    void log_status();

public:
    ARFramework(
            GraphManager&,
            grid::region,
            grid::point,
            grid::point,
            grid::region,
            std::function<bool(grid::point const&)> const&,
            grid::verification_engine_type_t const&,
            grid::region_abstraction_strategy_t const& 
                = grid::centralPointRegionAbstraction,
            grid::region_refinement_strategy_t const&
                = grid::HierarchicalDimensionRefinementStrategy(
                        grid::randomDimSelection,
                        2u,
                        5u
                    )
            );

    void set_verification_engine(
            grid::verification_engine_type_t const& v)
    { verification_engine = v; }
    void set_refinement_strategy(
            grid::region_refinement_strategy_t const& r)
    { refinement_strategy = r; }
    void set_abstraction_strategy(
            grid::region_abstraction_strategy_t const& a)
    { abstraction_strategy = a; }

    void run();
    void join() { keep_working = false; }

    template <class CallbackFunc>
    inline void report(CallbackFunc&& cb)
    {
        log_status();
        for(auto&& adv_exp_pair : unsafeRegionsWithAdvExamples)
        {
            cb(adv_exp_pair.second);
        }
    }
};

#endif

