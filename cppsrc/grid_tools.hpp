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
    using norm_function_type_t = std::function<double(point const&, point const&)>;

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
        IntellifeatureDimSelection(std::vector<point> const&, norm_function_type_t, std::size_t);
        std::vector<std::size_t> operator()(region const&, std::size_t);
        const std::vector<point> averages;
    private:
        norm_function_type_t norm_func;
        std::size_t orig_class;
    };

    struct IntelliFGSMRegionAbstraction
    {
        IntelliFGSMRegionAbstraction(std::size_t, std::function<point(point const&)>&&);
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

namespace std
{
    grid::point abs(grid::point const&);
}

std::ostream& operator<<(std::ostream& os, grid::region const&);
std::ostream& operator<<(std::ostream& os, grid::point const&);

template <class ArrayLike>
std::vector<std::size_t> getSortedIndices(ArrayLike const& arr, bool asc = true)
{
    std::vector<std::size_t> retInd(arr.size());
    std::iota(retInd.begin(), retInd.end(), 0u);
    std::sort(retInd.begin(), retInd.end(), [&arr, asc](std::size_t a, std::size_t b)
            {
                return asc ? arr[a] < arr[b] : arr[a] > arr[b];
            });
    return retInd;
}

#endif

