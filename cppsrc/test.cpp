#include "grid_tools.hpp"

#include <cassert>
#include <iostream>

bool pointInRegion(grid::point const& p, grid::region const& r)
{
    return !(p < r) && !(r < p);
}

bool regionEqual(grid::region const& r1, grid::region const& r2)
{
    return !(r1<r2) && !(r2<r1);
}


int main()
{
}
