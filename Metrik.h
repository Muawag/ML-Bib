#pragma once
#include <iostream>
#include <vector>
#include "Matrix.h"
#include <stdexcept>
#include "DataMatrix.h"

class Metrik {
    public:
        static double mse(const DataMatrix& d1, const DataMatrix& d2);
        static double mse(const Matrix& m1, const Matrix& m2);
        static double rmse(const DataMatrix& d1, const DataMatrix& d2);
        static double rmse(const Matrix& m1, const Matrix& m2);
        static double mae(const DataMatrix& d1, const DataMatrix& d2);
        static double mae(const Matrix& m1, const Matrix& m2);
};