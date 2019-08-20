#include <functional>
#include <iostream>
#include <numeric>
#include <vector>
#include <utility>

std::vector<std::pair<int,int>> vec = {{2,4},{1,7},{9,12}};
bool func(std::function<int(int)>&& f, int i)
{
    std::cout << f(i) << "\n";
    if(f(i) < 5) return false;
    return true;
}

int internal(int i)
{
    std::cout << "inside internal\n";
    return i+5;
}

auto bound = std::bind(&func, &internal, std::placeholders::_1);

int main()
{
    std::vector<int> vec2 = {4,8,5,1,6,4,2,7,5,9};
    std::vector<std::size_t> indices(vec2.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&vec2](std::size_t a, std::size_t b)
            {
                return vec2[a] > vec2[b];
            });
    for(auto&& index : indices) { std::cout << vec2[index] << " "; } std::cout << "\n";
    std::vector<std::size_t> a = {indices.begin(), indices.begin()+4};
    for(auto&& index : a) { std::cout << vec2[index] << " "; } std::cout << "\n";
    auto t = std::accumulate(vec.begin(), vec.end(), 1, [](int a, std::pair<int,int> const& elem)
            {
            return a*(elem.second - elem.first);
            });
    std::cout << t << "\n";
    return 0;
};
