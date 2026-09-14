#pragma once
#include <vector>
#include <iostream>
#include "SVD.h"
#include "Matrix.h"

class PCA {
    public:
        PCA();
        void train(const Matrix& m, double threshold = 0.95);
        Matrix apply(const Matrix& m) const;
        Matrix train_apply(const Matrix& m, double threshold = 0.95); 
    private:
        std::vector<double> means_;
        Matrix V_k;
        bool trained;
        std::vector<double> cum_var;
        int k;
};