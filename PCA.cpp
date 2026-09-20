#include "PCA.h"
#include <stdexcept>

std::vector<double> get_Spaltenmittelwerte(const Matrix& m);
Matrix zentrieren(const Matrix& m , const std::vector<double>& means);
std::vector<double> cumulative_variance(const Matrix& m);
std::size_t get_k(const std::vector<double>& cumsum, double threshold);

PCA::PCA() {
    trained = false;
    V_k = Matrix::identity(1);
}

void PCA::train(const Matrix& m, double threshold) {
    Matrix train_matrix = m;
    //means_ = get_Spaltenmittelwerte(m);
    //train_matrix = zentrieren(train_matrix, means_);
    SVD_Matrizen matricies = SVD::svd(train_matrix);
    cum_var = cumulative_variance(matricies.Sigma);
    k = get_k(cum_var, threshold);
    V_k = matricies.V.resize({matricies.V.get_Size().rows, k});
    trained = true;
}

Matrix PCA::apply(const Matrix& m) const{
    //Matrix zentriert = zentrieren(m,means_);
    Matrix result = m * V_k;
    return result;
}

Matrix PCA::train_apply(const Matrix& m, double threshold) {
    train(m, threshold);
    return apply(m);
}

std::vector<double> get_Spaltenmittelwerte(const Matrix& m) {
    std::vector<double> means;
    for(std::size_t j = 0; j < m.get_Size().cols; ++j) {
        double x = 0.0;
        for(std::size_t i = 0; i < m.get_Size().rows; ++i) {
            x += m[i][j];
        }
        means.push_back(x/m.get_Size().rows);
    }
    return means;
}

Matrix zentrieren(const Matrix& m , const std::vector<double>& means) {
    if(m.get_Size().cols != means.size()) {
        throw std::runtime_error("Mittelwertvektor hat nicht die grösse der spalten der Matrix");
    }
    Matrix_Size size = m.get_Size();
    Matrix result(m.get_Size());
    for(std::size_t i = 0; i < size.rows; ++i) {
        for(std::size_t j = 0; j < size.cols; ++j) {
            result[i][j] = m[i][j] - means[j];
        }
    }
    return result;
}

std::vector<double> cumulative_variance(const Matrix& m) {
    double total = 0.0;
    std::vector<double> singular_values = m.get_Diagonal_Vector();
    for(auto s : singular_values) total += s * s;
    
    std::vector<double> result;
    double cumsum = 0.0;
    for(auto s : singular_values) {
        cumsum += s * s / total;
        result.push_back(cumsum);
    }
    return result;
}

std::size_t get_k(const std::vector<double>& cumsum, double threshold) {
    for(std::size_t k = 0; k < cumsum.size(); ++k) {
        if(cumsum[k] >= threshold) {
            return k + 1;  
        }
    }
    return cumsum.size();  
}