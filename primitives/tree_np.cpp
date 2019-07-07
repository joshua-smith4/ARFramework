#include <iostream>
#include <algorithm>
#include <functional>
#include <set>
#include <utility>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace np = boost::python::numpy;
namespace py = boost::python;

template <class T>
class Tree
{
private:
    PyObject * compFunction;
    std::set<T,std::function<bool(T const&,T const&)>> a;
public:
    Tree(PyObject* p) 
        : compFunction(p), a([this](T const& a, T const& b){ return py::call<bool>(this->compFunction,a,b); })
    {
    }
    void insert(T const& p)
    {
        a.insert(p);
    }
    void erase(T const& p)
    {
        a.erase(p);
    }
    unsigned int size()
    {
        return a.size();
    }
    void for_each(PyObject* p)
    {
        std::for_each(a.begin(),a.end(),[p](T const& a){ py::call<void>(p,a); });
    }
    auto exists(T const& p)
    {
        return a.end() == a.find(p) ? false : true;
    }
    py::tuple find(T const& p)
    { 
        auto retVal = a.find(p);
        if(retVal == a.end()) return py::make_tuple(false);
        return py::make_tuple(true, *retVal);
    }
    py::tuple findAndErase(T const& p)
    {
        auto retVal = a.find(p);
        if(retVal == a.end()) return py::make_tuple(false);
        auto tmp = *retVal;
        a.erase(retVal);
        return py::make_tuple(true, tmp);
    }
    void clear()
    {
        a.clear();
    }
};    

BOOST_PYTHON_MODULE(tree)
{
    Py_Initialize();
    np::initialize();
    py::class_<Tree<np::ndarray>>("Tree", py::init<PyObject*>())
        .def("insert", &Tree<np::ndarray>::insert)
        .def("erase", &Tree<np::ndarray>::erase)
        .def("size", &Tree<np::ndarray>::size)
        .def("for_each", &Tree<np::ndarray>::for_each)
        .def("exists", &Tree<np::ndarray>::exists)
        .def("find", &Tree<np::ndarray>::find)
        .def("findAndErase", &Tree<np::ndarray>::findAndErase)
        .def("clear", &Tree<np::ndarray>::clear)
    ;
}

