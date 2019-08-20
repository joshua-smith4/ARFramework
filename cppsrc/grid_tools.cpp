#include <exception>
#include <stdexcept>
#include <numeric>

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

std::set<grid::point> centralPointRegionAbstraction(grid::region const& r)
{
    grid::point p(r.size());
    for(auto i = 0u; i < r.size(); ++i)
    {
        p[i] = (range.first + range.second) / static_cast<grid::numeric_type_t>(2.0);
    }
    return {p};
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
    std::vector<std::size_t> indices(p.size());
    for(auto i = 0u; i < p.size(); ++i)
    {
        dif[i] = std::abs(avg - elem);
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [&dif](std::size_t a, std::size_t b)
            {
                return dif[a] > dif[b];
            });

    return {indices.begin(), indices.begin() + numDims};
}

grid::IntellifeatureDimSelection::IntellifeatureDimSelection()
    : averages()
{
}

std::vector<std::size_t> 
grid::IntellifeatureDimSelection::operator()(grid::region const& r, std::size_t numDims)
{
    if(numDims > r.size())
        throw std::domain_error("Number of selected dimensions needs to match the dimensionality of the input")
    auto centralPointSet = grid::centralPointRegionAbstraction(r);
    if(centralPointSet.empty()) return {};
    auto p = *centralPointSet.begin();
    std::vector<double> norms(averages.size());
    for(auto i = 0u; i < averages.size(); ++i)
    {
        if(i == orig_class) continue;
        norms[i] = norm_func(averages[i], p);
    }
    std::vector<std::size_t> indices(averages.size());
    std::sort(indices.begin(), indices.end(), [&norms](std::size_t a, std::size_t b)
            {
                return norms[a] < norms[b];
            });
    if(indices.back() != orig_class)
    {
        std::remove(indices.begin(), indices.end(), orig_class);
        indices.push_back(orig_class);
    }
    auto minDiff = p - averages[indices[0]];
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
        if(p[i] > r[i].second) return false;
    }
    return false;
}

bool operator<(grid::region const& r, grid::point const& p)
{
    if(p.size() != r.size()) 
        throw std::domain_error("Point and region must have the same dimensionality.");
    for(auto i = 0u; i < p.size(); ++i)
    {
        if(r[i].second < p[i]) return true;
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
