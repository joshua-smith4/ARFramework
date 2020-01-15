#include "ARFramework.hpp"
#include <chrono>

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
        adversarialExamples(),
        ae_mutex(),
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
    /*
    for(auto&& row : orig_region)
        std::cout << row.first << " " << row.second << "\n";
    */
    if(!grid::isValidRegion(orig_region))
    {
        LOG(ERROR) << "Invalid original region";
        exit(1);
    }
    //potentiallyUnsafeRegions.push_back(orig_region);
    potentiallyUnsafeRegions.insert(orig_region);
}

void ARFramework::log_status()
{
    std::cout << "Unverified Regions: " 
        << potentiallyUnsafeRegions.size() << "\n";
    std::cout << "Unsafe Regions: " 
        << unsafeRegionsWithAdvExamples.size() << "\n";
    std::cout << "Adversarial Examples: "
        << adversarialExamples.size() << "\n";
    std::cout << "Safe Regions: " << safeRegions.size() << "\n";
}

void ARFramework::worker_routine()
{
    auto counter = 0u;
    while(keep_working)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        while(
                (!potentiallyUnsafeRegions.empty() || 
                 !unsafeRegionsWithAdvExamples.empty()) 
                && keep_working)
        {
            if(counter >= 100 && std::this_thread::get_id() == logging_thread_id)
            {
                log_status();
                counter = 0u;
            }
            ++counter;
            if(!potentiallyUnsafeRegions.empty())
            {
                grid::region selected_region;
                bool got_valid_region = false;
                {
                    std::lock_guard<std::mutex> lock(pur_mutex);
                    if(!potentiallyUnsafeRegions.empty())
                    {
                        // stack
                        /*
                        selected_region = potentiallyUnsafeRegions.back();
                        potentiallyUnsafeRegions.pop_back();
                        */

                        // queue
                        /*
                        selected_region = potentiallyUnsafeRegions.front();
                        potentiallyUnsafeRegions.pop_front();
                        */

                        selected_region = 
                            *potentiallyUnsafeRegions.begin();
                        potentiallyUnsafeRegions.erase(
                            potentiallyUnsafeRegions.begin());
                        got_valid_region = true;
                    }
                }
                if(!got_valid_region)
                {
                    continue;
                }
                selected_region = grid::snapToDomainRange(
                        selected_region,
                        domain_range);
                /*
                for(auto&& elem : granularity)
                    std::cout << elem << " ";
                std::cout << "\n";
                */
                unsigned long long numValidPoints = 
                    grid::AllValidDiscretizedPointsAbstraction
                    ::getNumberValidPoints(
                            selected_region,
                            init_point,
                            granularity);
                //std::cout << numValidPoints << std::endl;
                if(numValidPoints <= 0u) 
                {
                    continue;
                }
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
                    auto subregions = 
                        refinement_strategy(selected_region);
                    auto subregion_with_adv_exp =
                        subregions.find(verification_result.second);
                    if(subregions.end() == subregion_with_adv_exp)
                    {
                        LOG(ERROR) 
                            << "Adversarial example was found that did not belong to any subregions";
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
                    {
                        std::lock_guard<std::mutex> lock(pur_mutex);
                        /*
                        std::copy(subregions.begin(), subregions.end(),
                                std::back_inserter(potentiallyUnsafeRegions)
                                );
                        */
                        std::copy(subregions.begin(), subregions.end(),
                                std::inserter(potentiallyUnsafeRegions,
                                    potentiallyUnsafeRegions.begin())
                                );
                    }
                }
                else if(verification_result.first ==
                        grid::VERIFICATION_RETURN::UNKNOWN)
                {
                    auto subregions = refinement_strategy(selected_region);
                    std::map<grid::region, grid::point, grid::region_less_compare> unsafeRegionsTmp;
                    std::set<grid::point> all_abstracted_points;
                    auto first_iter = true;
                    for(auto&& subregion : subregions)
                    {
                        auto abstracted_points = 
                            abstraction_strategy(subregion);
                        if(first_iter)
                        {
                            auto abstraction_orig = 
                                abstraction_strategy(selected_region);
                            std::copy(abstraction_orig.begin(),
                                    abstraction_orig.end(),
                                    std::back_inserter(
                                        abstracted_points));
                            first_iter = false;
                        }
                        for(auto&& pt : abstracted_points)
                        {
                            auto discrete_pt = grid::enforceSnapDiscreteGrid(
                                    pt, init_point, granularity);
                            auto snapped_pt = grid::snapToDomainRange(
                                    discrete_pt,
                                    domain_range);
                            /*
                            auto snapped_pt = grid::snapToDomainRange(
                                    pt,
                                    domain_range);
                            */
                            if(grid::isInDomainRange(snapped_pt, domain_range))
                            {
                                /*
                                all_abstracted_points.push_back(
                                        snapped_pt);
                                */
                                all_abstracted_points.insert(
                                        snapped_pt);
                            }
                            else
                            {
                                LOG(ERROR)
                                    << "Point was outside domain range\n";
                            }
                        }
                    }

                    std::set<grid::region, grid::region_less_compare> 
                        deleted_regions;
                    for(auto&& pt : all_abstracted_points)
                    {
                        if(safety_predicate(pt)) continue;
                        if(!grid::isInDomainRange(pt, orig_region))
                            std::cout << "outside orig_region\n";
                        /*
                        auto in_deleted = deleted_regions.find(pt);
                        if(deleted_regions.end() != in_deleted)
                            continue;
                        */
                        {
                            std::lock_guard<std::mutex> lock(ae_mutex);
                            adversarialExamples.insert(pt);
                        }
                        auto found_subregion = subregions.find(pt);
                        if(subregions.end() != found_subregion)
                        {
                            unsafeRegionsTmp.insert(
                                    {*found_subregion, pt});
                            deleted_regions.insert(*found_subregion);
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
                                //std::cout << "found in pot unsafe regions\n";
                            }
                            else
                            {
                                //std::cout << "not found anywhere\n";
                            }
                        }
                    }
                    {
                        std::lock_guard<std::mutex> lock(ur_mutex);
                        std::copy(unsafeRegionsTmp.begin(),
                                unsafeRegionsTmp.end(),
                                std::inserter(
                                    unsafeRegionsWithAdvExamples,
                                    unsafeRegionsWithAdvExamples.begin())
                                );
                    }
                    {
                        std::lock_guard<std::mutex> lock(pur_mutex);
                        /*
                        std::copy(subregions.begin(),
                                subregions.end(),
                                std::back_inserter(potentiallyUnsafeRegions)
                                );
                        */
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
                {
                    std::lock_guard<std::mutex> lock(pur_mutex);
                    /*
                    std::copy(nonempty_subregions.begin(), 
                            nonempty_subregions.end(), 
                            std::back_inserter(potentiallyUnsafeRegions)
                            );
                    */
                    std::copy(nonempty_subregions.begin(), 
                            nonempty_subregions.end(), 
                            std::inserter(potentiallyUnsafeRegions, 
                                potentiallyUnsafeRegions.begin()));
                }
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
