#include <utility>
#include <iostream>
#include <string>
#include <functional>

class Tmp
{
public:
    template <class Func, class... Args>
    typename std::result_of<Func(Args...)>::type func(Func&& f, Args&&... args)
    {
        return f(std::forward<Args>(args)...);
    }
};

int main()
{
    Tmp a;
    auto b = a.func([](std::string const& a)
            {
                return a.size();
            }, "jimbob");
    std::cout << b << "\n";
    return 0;
}
