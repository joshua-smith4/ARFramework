#ifndef GRID_TOOLS_INCLUDED
#define GRID_TOOLS_INCLUDED

#include <vector>
#include <set>
#include <utility>
#include <ostream>
#include <functional>

namespace grid
{
    using numeric_type_t = long double;
    using region = std::vector<std::pair<numeric_type_t, numeric_type_t>>;
    using point = std::vector<numeric_type_t>;


    using refinement_strategy_return_t = std::set<region>;
    using region_refinement_strategy_t = 
        std::function<refinement_strategy_return_t(region const&)>;
    using region_filter_strategy_t = std::function<bool(region const&)>;
    using abstraction_strategy_return_t = std::set<point>;
    using region_abstraction_strategy_t = 
        std::function<abstraction_strategy_return_t(region const&)>;
    using dimension_selection_strategy_t = std::function<std::vector<std::size_t>(region const&, std::size_t)>;
    using norm_function_type_t = std::function<double(point const&, point const&)>;

    bool isValidRegion(region const&);

    struct VolumeThresholdFilterStrategy
    {
        explicit VolumeThresholdFilterStrategy(double);
        const numeric_type_t threshold;
        bool operator()(region const&);
    };

    abstraction_strategy_return_t 
    centralPointRegionAbstraction(region const&);

    struct AllValidDiscretizedPointsAbstraction
    {
        AllValidDiscretizedPointsAbstraction();
        grid::point knownValidPoint;
        grid::point granularity;
        abstraction_strategy_return_t operator()(grid::region const&);
        unsigned long long getNumberValidPoints(grid::region const&);
        std::pair<bool, grid::point> findValidPointInRegion(
                grid::region const&);
    private:
        void enumerateAllPoints(
                abstraction_strategy_return_t&, 
                grid::point&,
                std::size_t,
                grid::region const&);
    };

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
        abstraction_strategy_return_t operator()(region const&);
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

