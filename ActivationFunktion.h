#pragma once

#include <vector>
#include <cmath>
#include <functional>

struct ActivationFunktion {
    using Activation = std::function<double(double)>;
    
    inline static Activation Relu = [](const double val) {return std::max(0.0, val);};
    inline static Activation Sigmoid = [](const double val) {return 1.0/(1 + std::exp(-val));};
    inline static Activation LeakyRelu = [](const double val) { if(val < 0.0) return 0.01 * val; return val;};
    inline static Activation Linear = [](const double val) {return val;}; 
    inline static Activation Gelu = [](const double val, const Activation func = Sigmoid) { return val * func(val);};


};



