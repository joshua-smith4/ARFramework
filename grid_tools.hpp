#ifndef GRID_TOOLS_INCLUDED
#define GRID_TOOLS_INCLUDED

#include <random>
#include <vector>
#include <set>
#include <utility>
#include <ostream>
#include <functional>

namespace grid
{
    struct region_less_compare;
    // numeric type used to represent regions and points
    // long double for highest precision
    using numeric_type_t = long double;

    using region_element = std::pair<numeric_type_t, numeric_type_t>;
    // hyperrectangular region where each element of the
    // underlying container contains a tuple of arguments
    // first: lower bound (inclusive)
    // second: upper bound (exclusive)
    using region = std::vector<region_element>;

    // container representing a point in the input space
    using point = std::vector<numeric_type_t>;


    using refinement_strategy_return_t = 
        std::set<region, region_less_compare>; 
    // takes a region and returns a partition of that region
    using region_refinement_strategy_t = 
        std::function<refinement_strategy_return_t(region const&)>;

    // returns true if a region should be removed
    using region_filter_strategy_t = std::function<bool(region const&)>;

    using abstraction_strategy_return_t = std::vector<point>;
    // abstracts a region to a set of discrete points
    using region_abstraction_strategy_t = 
        std::function<abstraction_strategy_return_t(region const&)>;

    // selects a set of dimensions (vector of indices) that
    // are more important or useful than others based on
    // a heuristic
    using dim_selection_strategy_return_t = std::vector<std::size_t>;
    using dimension_selection_strategy_t = 
        std::function<dim_selection_strategy_return_t(region const&, std::size_t)>;

    // function used to calculate the norm of a point
    using norm_function_type_t = 
        std::function<long double(point const&)>;

    // takes a point and enforces alignment to a discrete
    // grid
    using discrete_enforcement_function_type_t = 
        std::function<point(point const&, point const&, point const&)>;

    enum class VERIFICATION_RETURN
    {
        SAFE,
        UNSAFE,
        UNKNOWN
    };

    using verification_engine_return_t = 
        std::pair<VERIFICATION_RETURN, point>;
    using verification_engine_type_t = 
        std::function<verification_engine_return_t(region const&)>;

    region
    snapToDomainRange(
            region const&, /* region */
            region const&);

    point
    snapToDomainRange(
            point const&, /* point */
            region const& /* domain range */);

    bool isInDomainRange(point const&, region const&);
    bool isInDomainRange(region const&, region const&);

    // think about making this a pass by reference to avoid
    // excessive copying
    point 
    enforceSnapDiscreteGrid(
            point const& /* p */, 
            point const& /* referencePoint */, 
            point const& /* granularity */);

    // definition of valid region
    bool isValidRegion(region const&);

    bool pointIsInRegion(region const&, point const&);

    long double regionVolume(region const&);
    // filter strategy based on the 'volume' of a region
    // compared to a threshold
    struct VolumeThresholdFilterStrategy
    {
        explicit VolumeThresholdFilterStrategy(numeric_type_t);
        const numeric_type_t threshold;
        bool operator()(region const&);
    };

    struct RandomPointRegionAbstraction
    {
        explicit RandomPointRegionAbstraction(unsigned);
        abstraction_strategy_return_t operator()(region const&);
        unsigned numPoints;
        std::minstd_rand0 generator;
    };

    // abstracts a region to the central point
    abstraction_strategy_return_t 
    centralPointRegionAbstraction(region const&);

    // abstracts a region to a covering set of discrete
    // points aligned with a grid defined by a 
    // reference point and a granularity
    // (discrete units of each dimension)
    struct AllValidDiscretizedPointsAbstraction
    {
        AllValidDiscretizedPointsAbstraction(
                point /* knownValidPoint */, 
                point /* granularity */);
        abstraction_strategy_return_t operator()(grid::region const&);
        static unsigned long long getNumberValidPoints(
                grid::region const&, 
                grid::point const&,
                grid::point const&);
        unsigned long long getNumberValidPoints(grid::region const&);
        static std::pair<bool, grid::point> findValidPointInRegion(
                grid::region const&, 
                grid::point const&,
                grid::point const&);
        std::pair<bool, grid::point> findValidPointInRegion(
                grid::region const&);
    private:
        bool enumerateAllPoints(
                abstraction_strategy_return_t&, 
                grid::point const&,
                std::size_t,
                grid::region const&);
        grid::point knownValidPoint;
        grid::point granularity;
    };

    struct DiscreteSearchVerificationEngine
    {
        DiscreteSearchVerificationEngine(
                std::function<bool(region const&)> const& /* should attempt? */,
                region_abstraction_strategy_t const& /* point generation */,
                std::function<bool(point const&)> const& /* is point safe */);
        verification_engine_return_t operator()(region const&);
    private:
        std::function<bool(region const&)> shouldAttemptCheck;
        region_abstraction_strategy_t discretePointGenerator;
        std::function<bool(point const&)> point_safe_func;
    };

    // random dimension selection algorithm
    dim_selection_strategy_return_t randomDimSelection(region const&, 
            std::size_t);
    dim_selection_strategy_return_t largestDimFirst(region const&, 
            std::size_t);

    // dimension selection algorithm outlined in
    // the DLV paper (Xiaowei et. al.)
    dim_selection_strategy_return_t maxAverageDimSelection(region const&, 
            std::size_t);

    numeric_type_t l2norm(point const&);

    // dimension selection algorithm defined in
    // IntelliFeatures paper (Joshua Smith et. al.)
    struct IntellifeatureDimSelection
    {
        IntellifeatureDimSelection(
                std::vector<point> const&, 
                norm_function_type_t, 
                std::size_t);
        std::vector<std::size_t> operator()(region const&, std::size_t);
        const std::vector<point> averages;
    private:
        norm_function_type_t norm_func;
        std::size_t orig_class;
    };

    point sign(point const&);

    struct ModifiedFGSMWithFallbackRegionAbstraction
    {
        ModifiedFGSMWithFallbackRegionAbstraction(
                std::size_t /* number of points to generate */, 
                std::function<point(point const&)> const& /* gradient */,
                dimension_selection_strategy_t const&,
                region_abstraction_strategy_t const&,
                point const&,
                double /* percent of dimensions for normal FGSM */);
        abstraction_strategy_return_t operator()(region const&);
    private:
        std::size_t maxPoints;
        std::function<point(point const&)> gradient;
        dimension_selection_strategy_t dim_select_strategy;
        region_abstraction_strategy_t fallback_strategy;
        const point granularity;
        double percentFGSM;
        std::minstd_rand0 rand_gen;
    };

    struct HierarchicalDimensionRefinementStrategy
    {
        HierarchicalDimensionRefinementStrategy(
                dimension_selection_strategy_t const&,
                unsigned /* dimension divisor */,
                unsigned /* number of dimensions to subdivide */);
        refinement_strategy_return_t operator()(region const&);
    private:
        bool enumerateAllRegions(
                refinement_strategy_return_t&,
                region const&,
                unsigned,
                dim_selection_strategy_return_t const& /* dims */,
                region const& orig_region);
        dimension_selection_strategy_t dim_select_strategy;
        unsigned dim_divisor;
        unsigned numDims;
    };
}

bool operator<(grid::point const&, grid::region const&);
bool operator<(grid::region const&, grid::point const&);
bool operator<(grid::region const&, grid::region const&);
bool operator<(grid::point const&, grid::point const&);

namespace grid
{
    struct region_less_compare
    {
        using is_transparent = void;
        bool operator()(grid::point const& p, grid::region const& r) const
        {
            return p < r;
        }
        bool operator()(grid::region const& r, grid::point const& p) const
        {
            return r < p;
        }
        bool operator()(grid::region const& r1, grid::region const& r2) const
        {
            return r1 < r2;
        }
        bool operator()(grid::point const& p1, grid::point const& p2) const
        {
            return p1 < p2;
        }
    };
}

grid::point operator-(grid::point const&, grid::point const&);
grid::point operator+(grid::point const&, grid::point const&);
grid::point elementWiseMult(grid::point const&, grid::point const&);

template <class T>
grid::point constVecMult(T const& t, grid::point const& p)
{
    grid::point ret(p.size());
    for(auto i = 0u; i < p.size(); ++i)
        ret[i] = t * p[i];
    return ret;
}

namespace std
{
    grid::point abs(grid::point const&);
}

std::ostream& operator<<(std::ostream& os, grid::region const&);
std::ostream& operator<<(std::ostream& os, grid::point const&);
std::ostream& operator<<(std::ostream& os, 
        grid::VERIFICATION_RETURN const&);


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

