#include "grid_tools.hpp"

#include <cmath>
#include <cassert>
#include <iostream>

int main()
{
    grid::point granularity({0.25, 1.25, 0.5});
    grid::point valid_point({0, 1, 2});
    grid::point close_point({1.2, -0.2, 2.25});
    grid::region reg({{1,3},{2,6},{1,5}});
    auto snapped_point = 
        grid::enforceSnapDiscreteGrid(
                close_point, 
                valid_point, 
                granularity);
    for(auto i = 0u; i < snapped_point.size(); ++i)
    {
        auto dif = (snapped_point[i] - valid_point[i]) / granularity[i];
        assert(dif == ceil(dif) && dif == round(dif));
    }
    assert(snapped_point[0] == 1.25);
    assert(snapped_point[1] == -0.25);
    assert(snapped_point[2] == 2.5);

    auto tf1 = grid::VolumeThresholdFilterStrategy(31);
    auto tf2 = grid::VolumeThresholdFilterStrategy(33);
    assert(!tf1(reg));
    assert(tf2(reg));

    auto pvec = *grid::centralPointRegionAbstraction(reg).begin();
    for(auto i = 0u; i < reg.size(); ++i)
    {
        assert(pvec[i] > reg[i].first && pvec[i] < reg[i].second);
    }
    assert(pvec[0] == 2);
    assert(pvec[1] == 4);
    assert(pvec[2] == 3);

    auto avd_abstr = grid::AllValidDiscretizedPointsAbstraction(
            valid_point, granularity);
    auto all_valid_points = avd_abstr(reg);
    auto num_valid_points = avd_abstr.getNumberValidPoints(reg);
    auto p_in_region = avd_abstr.findValidPointInRegion(reg);
    assert(p_in_region.first);
    assert(p_in_region.second[0] == 1);
    assert(p_in_region.second[1] == 2.25);
    assert(p_in_region.second[2] == 1);
    assert(all_valid_points.size() == num_valid_points);
    assert(num_valid_points == 192);

    auto dims_mad = grid::maxAverageDimSelection(reg, 3);
    assert(dims_mad.size() == 3);
    assert(dims_mad[0] == 0 && dims_mad[1] == 1 && dims_mad[2] == 2);

    auto reg_intelli = grid::region({{0.8,1},{-1,-0.8},{-1,-0.8}});
    auto orig_class = 3;
    auto averages = std::vector<grid::point>({
            {1, 1, 1},
            {1, 1,-1},
            {1,-1, 1},
            {1,-1,-1},
            {-1,1, 1},
            {-1,1,-1},
            {-1,-1,1},
            {-1,-1,-1}});
    auto intellifeature = grid::IntellifeatureDimSelection(
            averages, grid::l2norm, orig_class);
    auto dims_intelli = intellifeature(reg_intelli, reg_intelli.size());
    assert(dims_intelli.size() == reg_intelli.size());
    
    auto sign_test_point = grid::point({-0.001, 0, 0.001, 2, -2});
    auto sign_res = grid::sign(sign_test_point);
    for(auto&& elem : sign_res)
        assert(std::abs(elem) <= 1);
    assert(
            sign_res[0] == -1 && 
            sign_res[1] == 0 && 
            sign_res[2] == 1 &&
            sign_res[3] == 1 &&
            sign_res[4] == -1);
    
    // TODO: test IntelliFGSM with real model
    return 0;
}

