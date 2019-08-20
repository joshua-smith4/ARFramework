#ifndef GRID_TOOLS_INCLUDED
#define GRID_TOOLS_INCLUDED

#include <vector>
#include <set>
#include <utility>
#include <ostream>
#include <functional>

namespace grid
{
    using numeric_type_t = double;
    using region = std::vector<std::pair<numeric_type_t, numeric_type_t>>;
    using point = std::vector<numeric_type_t>;

    using region_refinement_strategy_t = std::function<std::set<region>(region const&, std::vector<numeric_type_t> const&)>;
    using region_filter_strategy_t = std::function<bool(region const&)>;
    using region_abstraction_strategy_t = std::function<std::set<point>(region const&)>;
    using dimension_selection_strategy_t = std::function<std::vector<std::size_t>(region const&, std::size_t)>;

    bool isValidRegion(region const&);

    struct VolumeThresholdFilterStrategy
    {
        explicit VolumeThresholdFilterStrategy(double);
        const numeric_type_t threshold;
        bool operator()(region const&);
    };

    std::set<grid::point> centralPointRegionAbstraction(region const&);

    std::vector<std::size_t> maxAverageDimSelection(region const&, std::size_t);

    struct IntellifeatureDimSelection
    {
        std::vector<point> averages;
    };

    struct IntelliFGSMRegionAbstraction
    {
        FGSMRegionAbstraction(std::size_t, std::function<point(point const&)>&&);
        std::set<point> operator()(region const&);
        std::size_t maxPoints;
        std::function<point(point const&)> gradient;
    };
}

bool operator<(grid::point const&, grid::region const&);
bool operator<(grid::region const&, grid::point const&);
bool operator<(grid::region const&, grid::region const&);
bool operator<(grid::point const&, grid::point const&);

grid::point operator-(grid::point const&, grid::point const&);
grid::point operator+(grid::point const&, grid::point const&);

std::ostream& operator<<(std::ostream& os, grid::region const&);
std::ostream& operator<<(std::ostream& os, grid::point const&);


#endif

