#include "LinearRegression.h"
#include "SVD.h"
#include "stdexcept"


void LinearRegression::train(const Matrix& x_daten, const Matrix& y_daten) {
    Matrix A = x_daten;
    auto[Q, R] = SVD::q_r_Decomposition(A);
    Matrix y_final = Q.transpose() * y_daten;
    params.resize(R.get_Size().cols);
    for(int i = R.get_Size().cols - 1; i >= 0; --i) {
        double sum = y_final[i][0];
        for(int j = i + 1; j < R.get_Size().cols; ++j) {
            sum -= params[j] * R[i][j];
        }
        params[i] = sum / R[i][i];
    }
    for(double d : params) {
        std::cout << d << std::endl;
    }
}

Matrix LinearRegression::predict(const Matrix& predict_daten) const {
    if(params.size() != predict_daten.get_Size().cols) throw std::runtime_error("Matrix ist nicht passend zu den trainierten daten");
    std::vector<double> ret_vec;
    for(std::size_t i = 0; i < predict_daten.get_Size().rows; ++i) {
        double predict = 0.0;
        for(std::size_t j = 0; j < params.size(); ++j) {
            predict += params[j] * predict_daten[i][j];
        }
        ret_vec.push_back(predict);
    }
    Matrix result({ret_vec});
    return result.transpose();
}