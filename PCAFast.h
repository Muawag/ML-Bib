#pragma once

#include "Matrix.h"
#include "SVD.h"

class PCAFast {
    public:
        PCAFast();
        void train(const Matrix& m, double threshold = 0.95);
        Matrix apply(const Matrix& m) const;
        Matrix train_apply(const Matrix& m, double threshold = 0.95); 
    private:
        Matrix V_k;
        bool trained;
        int k;
        bool flat;
};