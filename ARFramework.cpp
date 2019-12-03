#include "ARFramework.hpp"

ARFramework::ARFramework(
        GraphManager& graph_manager,
        grid::region dr,
        grid::point ip,
        grid::point gran,
        grid::region orig_region,
        std::function<bool(grid::point const&)> const& safety_pred,
        grid::verification_engine_type_t const& verif_engine, 
        grid::region_abstraction_strategy_t const& abs_strat, 
        grid::region_refinement_strategy_t const& ref_strat)
    : 
        potentiallyUnsafeRegions(),
        pur_mutex(),
        safeRegions(),
        sr_mutex(),
        unsafeRegionsWithAdvExamples(),
        ur_mutex(),
        keep_working(true),
        gm(graph_manager),
        domain_range(dr),
        init_point(ip),
        granularity(gran),
        abstraction_strategy(abs_strat),
        refinement_strategy(ref_strat),
        verification_engine(verif_engine),
        safety_predicate(safety_pred),
        logging_thread_id(),
        log_thread_set(ATOMIC_FLAG_INIT)
{
    if(!gm.ok()) exit(1);
    orig_region = grid::snapToDomainRange(orig_region, domain_range);
    if(!grid::isValidRegion(orig_region))
    {
        LOG(ERROR) << "Invalid original region";
        exit(1);
    }
    potentiallyUnsafeRegions.insert(orig_region);
}

void ARFramework::worker_routine()
{
    auto counter = 0u;
    while(keep_working)
    {
        while(
                (!potentiallyUnsafeRegions.empty() || 
                 !unsafeRegionsWithAdvExamples.empty()) 
                && keep_working)
        {
            if(counter >= 100 && std::this_thread::get_id() == logging_thread_id)
            {
                std::cout << "Potentially Unsafe Regions: " << potentiallyUnsafeRegions.size()
                    << "\n";
                std::cout << "Unsafe Regions: " << unsafeRegionsWithAdvExamples.size() << "\n";
                std::cout << "Safe Regions: " << safeRegions.size() << "\n";
                counter = 0u;
            }
            ++counter;
            if(!potentiallyUnsafeRegions.empty())
            {
                grid::region selected_region;
                {
                    std::lock_guard<std::mutex> lock(pur_mutex);
                    if(!potentiallyUnsafeRegions.empty())
                    {
                        selected_region = 
                            *potentiallyUnsafeRegions.begin();
                        potentiallyUnsafeRegions.erase(
                                potentiallyUnsafeRegions.begin());
                    }
                }
                if(selected_region.empty()) continue;
                selected_region = grid::snapToDomainRange(
                        selected_region,
                        domain_range);
                auto numValidPoints = 
                    grid::AllValidDiscretizedPointsAbstraction
                    ::getNumberValidPoints(
                            selected_region,
                            init_point,
                            granularity);
                if(numValidPoints <= 0) continue;
                auto verification_result = 
                    verification_engine(selected_region);
                if(verification_result.first ==
                        grid::VERIFICATION_RETURN::SAFE)
                {
                    std::lock_guard<std::mutex> lock(sr_mutex);
                    safeRegions.push_back(selected_region);
                }
                else if(verification_result.first ==
                        grid::VERIFICATION_RETURN::UNSAFE)
                {
                    auto subregions = refinement_strategy(selected_region);
                    auto subregion_with_adv_exp =
                        subregions.find(verification_result.second);
                    if(subregions.end() == subregion_with_adv_exp)
                    {
                        LOG(ERROR) << "Adversarial example was found that did not belong to any subregions";
                    }
                    else
                    {
                        {
                            std::lock_guard<std::mutex> lock(ur_mutex);
                            unsafeRegionsWithAdvExamples.insert({
                                    *subregion_with_adv_exp,
                                    verification_result.second
                                    });
                        }
                        subregions.erase(subregion_with_adv_exp);
                    }
                    std::lock_guard<std::mutex> lock(pur_mutex);
                    std::copy(subregions.begin(), subregions.end(),
                            std::inserter(potentiallyUnsafeRegions,
                                potentiallyUnsafeRegions.begin())
                            );
                }
                else if(verification_result.first ==
                        grid::VERIFICATION_RETURN::UNKNOWN)
                {
                    auto subregions = refinement_strategy(selected_region);
                    std::map<grid::region, grid::point, grid::region_less_compare> unsafeRegionsTmp;
                    std::vector<grid::point> all_abstracted_points;
                    for(auto&& subregion : subregions)
                    {
                        auto abstracted_points = 
                            abstraction_strategy(subregion);
                        for(auto&& pt : abstracted_points)
                        {
                            auto snapped_pt = grid::snapToDomainRange(
                                    grid::enforceSnapDiscreteGrid(
                                        pt,
                                        init_point,
                                        granularity),
                                    domain_range);
                            if(grid::isInDomainRange(snapped_pt, domain_range))
                            {
                                all_abstracted_points.push_back(
                                        snapped_pt);
                            }
                            else
                            {
                                LOG(ERROR)
                                    << "Point was outside domain range\n";
                            }
                        }
                    }
                    for(auto&& pt : all_abstracted_points)
                    {
                        if(safety_predicate(pt)) continue;
                        auto found_subregion = subregions.find(pt);
                        if(subregions.end() != found_subregion)
                        {
                            unsafeRegionsTmp.insert(
                                    {*found_subregion, pt});
                            subregions.erase(found_subregion);
                        }
                        else
                        {
                            std::lock_guard<std::mutex> lock(pur_mutex);
                            auto found_region = 
                                potentiallyUnsafeRegions.find(pt);
                            if(potentiallyUnsafeRegions.end() 
                                    != found_region)
                            {
                                unsafeRegionsTmp.insert(
                                        {*found_region, pt});
                                potentiallyUnsafeRegions.erase(
                                        found_region);
                            }
                        }
                    }
                    {
                        std::lock_guard<std::mutex> lock(ur_mutex);
                        std::copy(unsafeRegionsTmp.begin(),
                                unsafeRegionsTmp.end(),
                                std::inserter(unsafeRegionsWithAdvExamples,
                                    unsafeRegionsWithAdvExamples.begin()));
                    }
                    {
                        std::lock_guard<std::mutex> lock(pur_mutex);
                        std::copy(subregions.begin(),
                                subregions.end(),
                                std::inserter(potentiallyUnsafeRegions,
                                    potentiallyUnsafeRegions.begin()));
                    }
                }
            }
            else
            {
                grid::region selected_region;
                grid::point adv_exp;
                {
                    std::lock_guard<std::mutex> lock(ur_mutex);
                    if(!unsafeRegionsWithAdvExamples.empty())
                    {
                        auto tmp_iter = 
                            unsafeRegionsWithAdvExamples.begin();
                        selected_region = tmp_iter->first;
                        adv_exp = tmp_iter->second;
                        unsafeRegionsWithAdvExamples.erase(tmp_iter);
                    }
                }
                if(selected_region.empty() ||
                        grid::AllValidDiscretizedPointsAbstraction
                        ::getNumberValidPoints(
                            selected_region, 
                            init_point, 
                            granularity) 
                        <= 1ull)
                {
                    continue;
                }
                auto subregions = refinement_strategy(selected_region);
                std::set<grid::region, grid::region_less_compare>
                    nonempty_subregions;
                for(auto&& subregion : subregions)
                {
                    if(grid::AllValidDiscretizedPointsAbstraction
                            ::getNumberValidPoints(
                                subregion,
                                init_point,
                                granularity) > 0ull)
                    {
                        nonempty_subregions.insert(subregion);
                    }
                }
                if(nonempty_subregions.empty()) continue;
                auto unsafeRegionIter = nonempty_subregions.find(adv_exp);
                if(unsafeRegionIter != nonempty_subregions.end())
                {
                    {
                        std::lock_guard<std::mutex> lock(ur_mutex);
                        unsafeRegionsWithAdvExamples.insert(
                                {*unsafeRegionIter, adv_exp});
                    }
                    nonempty_subregions.erase(unsafeRegionIter);
                }
                else
                {
                    LOG(ERROR) << "Adv exp was found not belonging to region after refined";
                }
                std::lock_guard<std::mutex> lock(pur_mutex);
                std::copy(nonempty_subregions.begin(), 
                        nonempty_subregions.end(), 
                        std::inserter(potentiallyUnsafeRegions, 
                            potentiallyUnsafeRegions.begin()));
            }
        }
    }
}

void ARFramework::run()
{
    if(!log_thread_set.test_and_set(std::memory_order_acquire))
        logging_thread_id = std::this_thread::get_id();
    worker_routine();
}
