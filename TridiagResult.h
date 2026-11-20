#pragma once

#include<vector>
#include"Matrix.h"

    struct TridiagonalResult{
        std::vector<double> subdiagonal;
        std::vector<double> diagonal;
        std::vector<double> taus;
        Matrix reflectors;
    };

    struct EigenResult {
        std::vector<double> eigenwerte;
        Matrix vectors;
    };