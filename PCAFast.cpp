#include "PCAFast.h"

void PCAFast::train(const Matrix& m, double threshold) {

    auto [U, Sigma] = SVD::svd_gramm_for_pca(m, threshold, &flat);
    std::cout << "SVD" << std::endl;
    k = U.get_Size().cols;

    if (!flat) {
        V_k = U;                                
    } else {
        V_k = m.transpose() * U;               
        double* d = V_k.data();
        const std::size_t rows = V_k.get_Size().rows;
        const std::size_t cols = V_k.get_Size().cols;
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols; ++j)
                d[i * cols + j] /= Sigma[j];       
    }
    trained = true;
}

Matrix PCAFast::apply(const Matrix& m) const{
    if (!trained) throw std::logic_error("PCAFast: nicht trainiert");
    if (m.get_Size().cols != V_k.get_Size().rows)
        throw std::invalid_argument("PCAFast: falsche Spaltenzahl");
    return m * V_k;
}

Matrix PCAFast::train_apply(const Matrix& m, double threshold) {
    train(m,threshold);
    return apply(m);
}

PCAFast::PCAFast() {
    trained = false;
    V_k = Matrix::identity(1);
}

