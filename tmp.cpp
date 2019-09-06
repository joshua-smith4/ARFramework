#include <cmath>
#include <cassert>
#include <iostream>
#include <set>
#include <vector>
#include <algorithm>

using num_t = long double;
using region = std::vector<std::pair<num_t, num_t>>;
using point = std::vector<num_t>;

bool enumerateAllPoints(
        std::set<point> &s, 
        point const& p, 
        std::size_t curIndex,
        region const& r,
        point const& granularity)
{
    if(curIndex >= r.size()) return true;
    auto newp = p;
    while(newp[curIndex] < r[curIndex].second)
    {
        if(enumerateAllPoints(s, newp, curIndex+1, r, granularity))
            std::fill_n(std::inserter(s, s.end()), 1, newp);
        newp[curIndex] += granularity[curIndex];
    }
    return false;
}

unsigned long long 
getNumberValidPoints(
        region const& r,
        point const& vp,
        point const& gran)
{
    auto retVal = 1ull;
    for(auto i = 0u; i < r.size(); ++i)
    {
        if(r[i].first == r[i].second) continue;
        auto numPointsInDim =  
            static_cast<unsigned long long>(
                    ceil((r[i].second - vp[i])
                    / gran[i]));
        retVal *= numPointsInDim;
    }
    return retVal;
}

int main()
{
    std::set<point> points;
    point valid_point;
    region r;
    point granularity;

    r = {{0,4},{0,3},{0,2}};
    valid_point = {0.51, 2.9, 1.74};
    granularity = {0.5, 2, 0.25};

    enumerateAllPoints(points, valid_point, 0, r, granularity);

    std::cout << "Generated: " << points.size() << " points\n";
    assert(points.size() == getNumberValidPoints(r, valid_point, granularity));
    return 0;
}

