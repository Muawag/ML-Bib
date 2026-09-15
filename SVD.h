#pragma once
#include <iostream>
#include "Matrix.h"
#include "Q_R_Matrizen.h"

struct SVD_Matrizen {
    Matrix U, Sigma, V;
};

struct Bidiagonalisierte_Matrizen {
    Matrix P, J, Q;
};

struct Q_R_Teil {
    std::vector<double> vec;
    double val;
};

struct Givens_Matrizen {
    Matrix G_R, D, G_L;
};

class SVD {
    public:
        SVD() = default;
        static SVD_Matrizen svd(Matrix& m);
        static Matrix generate_Bidiagonalisierte_Matrix(const Matrix_Size& size, const std::vector<double>& as, const std::vector<double>& bs);
        static Bidiagonalisierte_Matrizen bidiagonalize(Matrix& m);
        static Q_R_Matrizen q_r_step_spalte(Matrix& m);
        static Q_R_Matrizen q_r_step_zeile(Matrix& m);
        static Matrix get_Matrix_From_Smaller(std::size_t size, const std::vector<Matrix>& matricies);
        static SVD_Matrizen get_SVD_From_Bidiagonal(Bidiagonalisierte_Matrizen& matricies);
        static Givens_Matrizen givens_Iteration(Matrix& m);
        static SVD_Matrizen clean_up_Svd(const Matrix_Size& original_size, SVD_Matrizen& matricies);
        static Q_R_Matrizen q_r_Decomposition(const Matrix& m);
        private:
            //Für spätere Attribute evnt;
};

