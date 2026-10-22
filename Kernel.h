#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <span>


struct LinearKernel {
    double operator()(std::span<const double> a, std::span<const double> b) const {
        if(a.size() != b.size()) throw std::runtime_error("Vektoren unterschiedlich gross");
        double result = 0.0; 
        for(std::size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
        return result;
    }
};

struct PolyKernel {
    double degree, coef0;
    PolyKernel(int degree = 3, double coef0 = 1.0) : degree(degree), coef0(coef0) {}
    double operator()(std::span<const double> a, std::span<const double> b) const {
        if(a.size() != b.size()) throw std::runtime_error("Vektoren unterschiedlich gross");
        double result = 0.0;
        for(std::size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
        return std::pow(result + coef0, degree);
    }
};

struct RBFKernel {
    double gamma;
    explicit RBFKernel(double gamma = 1.0) : gamma(gamma) {}
    double operator()(std::span<const double> a, std::span<const double> b) const {
        if(a.size() != b.size()) throw std::runtime_error("Vektoren unterschiedlich gross");
        double d2 = 0.0;
        for(std::size_t i = 0; i < a.size(); ++i) {
            double d = a[i] - b[i];
            d2 += d * d;
        }
        return std::exp(-gamma * d2);
    }
};