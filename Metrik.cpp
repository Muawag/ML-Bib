#include "Metrik.h"
#include <cmath>

double Metrik::mse(const Matrix& m1, const Matrix& m2) {
    size_t n1 = (m1.get_Size().rows == 1) ? m1.get_Size().cols : m1.get_Size().rows;
    size_t n2 = (m2.get_Size().rows == 1) ? m2.get_Size().cols : m2.get_Size().rows;
    
    if (n1 != n2)
        throw std::runtime_error("Vektoren muessen gleich gross sein");

    double sum = 0.0;
    for (size_t i = 0; i < n1; ++i) {
        double v1 = (m1.get_Size().rows == 1) ? m1[0][i] : m1[i][0];
        double v2 = (m2.get_Size().rows == 1) ? m2[0][i] : m2[i][0];
        sum += (v1 - v2) * (v1 - v2);
    }

    return sum / n1;
}

double Metrik::mse(const DataMatrix& d1, const DataMatrix& d2) {
    return mse(Matrix(d1),Matrix(d2));
}

double Metrik::rmse(const Matrix& m1, const Matrix& m2) {
    return std::sqrt(mse(m1,m2));
}

double Metrik::rmse(const DataMatrix& d1, const DataMatrix& d2) {
    return std::sqrt(mse(d1,d2));
}

double Metrik::mae(const Matrix& m1, const Matrix& m2) {
    size_t n1 = (m1.get_Size().rows == 1) ? m1.get_Size().cols : m1.get_Size().rows;
    size_t n2 = (m2.get_Size().rows == 1) ? m2.get_Size().cols : m2.get_Size().rows;
    
    if (n1 != n2)
        throw std::runtime_error("Vektoren muessen gleich gross sein");

    double sum = 0.0;
    for (size_t i = 0; i < n1; ++i) {
        double v1 = (m1.get_Size().rows == 1) ? m1[0][i] : m1[i][0];
        double v2 = (m2.get_Size().rows == 1) ? m2[0][i] : m2[i][0];
        sum += std::abs(v1 - v2);
    }

    return sum / n1;
}

double Metrik::mae(const DataMatrix& d1, const DataMatrix& d2) {
    return mae(Matrix(d1), Matrix(d2));
}

int Metrik::klassifikationsfehler(const DataMatrix& d1, const DataMatrix& d2) {
    if(d1.get_Size() != d2.get_Size() || !d1.is_Vector()) {
        throw std::runtime_error("Daten müssen gleich Gross sein und ein Vector"); 
    }
    std::vector<std::string> v_1 = d1.get_Raw()[0]->get_Daten_as_String();
    std::vector<std::string> v_2 = d2.get_Raw()[0]->get_Daten_as_String();

    int fehler = 0;

    for(std::size_t i = 0; i < v_1.size(); ++i) {
        fehler = (v_1[i] == v_2[i]) ? fehler : fehler + 1;
    }
    return fehler;
}