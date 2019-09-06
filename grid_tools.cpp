#include <exception>
#include <stdexcept>
#include <numeric>
#include <cmath>

#include "grid_tools.hpp"

bool grid::isValidRegion(grid::region const & r)
{
    for(auto&& range : r)
    {
        if(range.first > range.second) return false;
    }
    return true;
}

grid::VolumeThresholdFilterStrategy::VolumeThresholdFilterStrategy(grid::numeric_type_t t)
    : threshold(t)
{
}

bool grid::VolumeThresholdFilterStrategy::operator()(grid::region const& r)
{
    auto volume = std::accumulate(r.begin(), r.end(), static_cast<grid::numeric_type_t>(1.0), 
        [](grid::numeric_type_t acc, std::pair<grid::numeric_type_t,grid::numeric_type_t> const& elem)
            {
                return acc*(elem.second - elem.first);
            });
    return volume < threshold;
}

grid::abstraction_strategy_return_t centralPointRegionAbstraction(grid::region const& r)
{
    grid::point p(r.size());
    for(auto i = 0u; i < r.size(); ++i)
    {
        p[i] = (range.first + range.second) / static_cast<grid::numeric_type_t>(2.0);
    }
    return {p};
}

grid::AllValidDiscretizedPointsAbstraction::AllValidDiscretizedPointsAbstraction(grid::point vp, grid::point gran)
    : knownValidPoint(vp), granularity(gran)
{
}

grid::abstraction_strategy_return_t 
grid::AllValidDiscretizedPointsAbstraction::operator()(
        grid::region const& r)
{
    auto validPoint = findValidPointInRegion(r);
    if(!validPoint.first) return {};
    grid::abstraction_strategy_return_t retVal;
    enumerateAllPoints(retVal, validPoint.second);
    return retVal;
}

unsigned long long 
grid::AllValidDiscretizedPointsAbstraction::getNumberValidPoints(
        grid::region const& r)
{
    auto validPointInRegion = findValidPointInRegion(r);
    if(!validPointInRegion.first) return 0ull;
    auto retVal = 1ull;
    for(auto i = 0u; i < r.size(); ++i)
    {
        if(r[i].first == r[i].second) continue;
        auto numPointsInDim =  
            static_cast<unsigned long long>(
                    ceil((r[i].second - validPointInRegion.second[i])
                    / granularity[i]));
        retVal *= numPointsInDim;
    }
    return retVal;
}

std::pair<bool, grid::point> 
grid::AllValidDiscretizedPointsAbstraction::findValidPointInRegion(
        grid::region const& r)
{
    grid::point retVal(r.size());
    for(auto i = 0u; i < r.size(); ++i)
    {
        auto multiplier = 
            ceil((r[i].first - knownValidPoint[i]) / granularity[i]);
        auto value = knownValidPoint[i] + multiplier*granularity[i]; 
        if(value >= r[i].second && value != r[i].first)
            return {false, {}};
        retVal[i] = value;
    }
    return {true, retVal};
}

void grid::AllValidDiscretizedPointsAbstraction::enumerateAllPoints(
        grid::abstraction_strategy_return_t& s, 
        grid::point& p, 
        std::size_t curIndex,
        grid::region const& r)
{
    if(curIndex >= r.size() || p[curIndex] >= r[curIndex].second) return;
    std::fill_n(std::inserter(s, s.end()), 1, p);
    while(p[curIndex] < r[curIndex].second)
    {
        p[curIndex] += granularity[curIndex];
        enumerateAllPoints(s, p, curIndex+1, r);
    }
}

std::vector<std::size_t> maxAverageDimSelection(grid::region const& r, std::size_t numDims)
{
    if(numDims > r.size()) throw std::domain_error("Number of selected dims must be less than or equal to the number of dimensions of the input");
    auto centralPoint = centralPointRegionAbstraction(r);
    if(centralPoint.empty()) return {};
    auto p = *centralPoint.begin();
    auto avg = std::reduce(p.begin(), p.end(), static_cast<grid::numeric_type_t>(0))
        / static_cast<grid::numeric_type_t>(p.size());

    grid::point dif(p.size());
    for(auto i = 0u; i < p.size(); ++i)
    {
        dif[i] = std::abs(avg - elem);
    }
    auto indices = getSortedIndices(dif, false /* descending */);
    return {indices.begin(), indices.begin() + numDims};
}

grid::IntellifeatureDimSelection::IntellifeatureDimSelection(
        std::vector<point> const& avgs,
        norm_function_type_t norm,
        std::size_t orig_cl)
    : averages(avgs), norm_func(norm), orig_class(orig_cl)
{
}

std::vector<std::size_t> 
grid::IntellifeatureDimSelection::operator()(grid::region const& r, std::size_t numDims)
{
    if(numDims > r.size())
        throw std::domain_error("Number of selected dimensions needs to match the dimensionality of the input");
    // get central point of region
    auto centralPointSet = grid::centralPointRegionAbstraction(r);
    if(centralPointSet.empty()) return {};
    auto p = *centralPointSet.begin();
    // calculate the norms between the averages and the central point
    std::vector<double> norms(averages.size());
    for(auto i = 0u; i < averages.size(); ++i)
    {
        // skip the original class
        if(i == orig_class) continue;
        norms[i] = norm_func(averages[i], p);
    }
    auto indices = getSortedIndices(norms);
    std::remove(indices.begin(), indices.end(), orig_class);
    indices.push_back(orig_class);
    auto minDiff = std::abs(p - averages[indices[0]]);
    indices = getSortedIndices(minDiff, false /* descending */);
    return {indices.begin(), indices.begin() + numDims};
}

grid::IntelliFGSMRegionAbstraction::IntelliFGSMRegionAbstraction()
    : maxPoints(), gradient()
{
}

bool operator<(grid::point const& p, grid::region const& r)
{
    if(p.size() != r.size()) 
        throw std::domain_error("Point and region must have the same dimensionality.");
    for(auto i = 0u; i < p.size(); ++i)
    {
        if(p[i] < r[i].first) return true;
        if(p[i] >= r[i].second) return false;
    }
    return false;
}

bool operator<(grid::region const& r, grid::point const& p)
{
    if(p.size() != r.size()) 
        throw std::domain_error("Point and region must have the same dimensionality.");
    for(auto i = 0u; i < p.size(); ++i)
    {
        if(r[i].second <= p[i]) return true;
        if(r[i].first > p[i]) return false;
    }
    return false;
}

bool operator<(grid::region const& r1, grid::region const& r2)
{
    if(r1.size() != r2.size()) 
        throw std::domain_error("Regions must have the same dimensionality.");
    for(auto i = 0u; i < r1.size(); ++i)
    {
        if(r1[i].second <= r2[i].first) return true;
        if(r1[i].first >= r2[i].second) return false;
    }
    return false;
}

bool operator<(grid::point const& p1, grid::point const& p2)
{
    if(p1.size() != p2.size()) 
        throw std::domain_error("Points must have the same dimensionality.");
    for(auto i = 0u; i < p1.size(); ++i)
    {
        if(p1[i] < p2[i]) return true;
        if(p1[i] > p2[i]) return false;
    }
    return false;
}

grid::point operator-(grid::point const& a, grid::point const& b)
{
    grid::point retVal(a.size());
    for(auto i = 0u; i < a.size(); ++i)
    {
        retVal[i] = a[i] - b[i];
    }
    return retVal;
}

grid::point operator+(grid::point const& a, grid::point const& b)
{
    grid::point retVal(a.size());
    for(auto i = 0u; i < a.size(); ++i)
    {
        retVal[i] = a[i] + b[i];
    }
    return retVal;
}

grid::point std::abs(grid::point const& p)
{
    grid::point ret(p.size());
    for(auto i = 0u; i < p.size(); ++i)
        ret[i] = std::abs(p[i]);
    return ret;
}

std::ostream& operator<<(std::ostream& os, grid::region const& r)
{
    os << "| ";
    for(auto&& range : r) 
        os << "(" << range.first << "," << range.second << ")" << " | ";
    return os;
}

std::ostream& operator<<(std::ostream& os, grid::point const& p)
{
    os << "| ";
    for(auto&& r : p)
        os << "(" << r << ")" << " | ";
    return os;
}
