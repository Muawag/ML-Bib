#include "Utility.h"



    double householder(double* , int , double* );

    std::vector<std::pair<Matrix, Matrix>> Utility::get_Batches(const Matrix& x_vals, const Matrix& y_vals, int size) {
        if(size > x_vals.get_Size().rows) {
            std::cout << "Batchsize grösser als Trainigsdaten" << std::endl;
            return {{x_vals, y_vals}};
        }
        int batch_count = x_vals.get_Size().rows / size;
        std::vector<std::pair<Matrix, Matrix>> result;
        result.reserve(batch_count);
        std::vector<int> numbers;
        for(std::size_t i = 0; i < x_vals.get_Size().rows; ++i) {
            numbers.push_back(i);
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(numbers.begin(), numbers.end(), gen);
        for(std::size_t i = 0; i < batch_count; ++i) {
            Matrix x_Batches(size, x_vals.get_Size().cols);
            Matrix y_Batches(size, y_vals.get_Size().cols);
            for(std::size_t j = 0; j < size; ++j) {
                int offset = i * size + j;
                auto x_src = x_vals[numbers[offset]];
                std::copy(x_src.begin(), x_src.end(), x_Batches[j].begin());

                auto y_src = y_vals[numbers[offset]];
                std::copy(y_src.begin(), y_src.end(), y_Batches[j].begin());
            }
            result.push_back({x_Batches, y_Batches});
        }
        return result;
    }

    Matrix Utility::prepare_SVM_Matrix(const Matrix& y_Daten, const int index) {
        Matrix y_result(y_Daten.get_Size());
        for(std::size_t i = 0; i < y_Daten.get_Size().rows; ++i) {
            y_result[i][0] = (y_Daten[i][0] == index) ? 1: -1;
        }
        return y_result;
    }

    TridiagonalResult Utility::tridiagonalize_symmetric_Matrix(Matrix& matrix) {
        if(matrix.get_Size().rows < 3) throw std::invalid_argument("Matrix zu klein um zu tridiagonalisieren");
        // isngesamt n-2 iterationen
        
        std::size_t size = matrix.get_Size().rows;
        std::vector<double> sub_diagonal(size-2);
        std::vector<double> taus(size > 2 ? size - 2 : 0);
        std::vector<double> v(size), p(size), w(size);
        double* A = matrix.data();
        double a_i;
        for (std::size_t i = 0; i + 2 < size; ++i) {
            double* x = A + i * size + (i + 1);
            std::size_t len = size - i - 1;
            taus[i] = householder(x, static_cast<int>(len), &a_i);
            sub_diagonal[i] = a_i;
            double tau = taus[i];
            if (tau == 0.0) continue;

            v[0] = 1.0;
            for (std::size_t k = 1; k < len; ++k) v[k] = x[k];

            double* B = A + (i + 1) * size + (i + 1);

            for (std::size_t r = 0; r < len; ++r) p[r] = 0.0;
            for (std::size_t r = 0; r < len; ++r) {
                const double* row = B + r * size;
                p[r] += row[r] * v[r];
                for (std::size_t c = r + 1; c < len; ++c) {
                    p[r] += row[c] * v[c];
                    p[c] += row[c] * v[r];
                }
            }
            double pv = 0.0;
            for (std::size_t r = 0; r < len; ++r) { p[r] *= tau; pv += p[r] * v[r]; }

            for (std::size_t r = 0; r < len; ++r) w[r] = p[r] - 0.5 * tau * pv * v[r];

            for (std::size_t r = 0; r < len; ++r) {
                double* row = B + r * size;
                for (std::size_t c = r; c < len; ++c)
                    row[c] -= v[r] * w[c] + w[r] * v[c];
            }
        }
        std::vector<double> diag = matrix.get_Diagonal_Vector();
        return {sub_diagonal, diag, taus, std::move(matrix)};

    }

    double householder(double* x, int len, double* a) {
        double tail = 0.0;
        for (int i = 1; i < len; ++i) tail += x[i] * x[i];

        *a = x[0];
        if (tail == 0.0) return 0.0;              // H = I, x bleibt wie es ist

        double x0 = x[0];
        double alpha = -std::copysign(1.0, x0) * std::sqrt(x0 * x0 + tail);
        *a = alpha;
        double tau = (alpha - x0) / alpha;
        double scale = 1.0 / (x0 - alpha);

        for (int i = 1; i < len; ++i) x[i] *= scale;   // v[1:] direkt in der Matrix
        x[0] = alpha;                                  // Nebendiagonale e[i]
        return tau;
    }

    EigenResult implicit_QR_shift(std::vector<double>& diag, std::vector<double>& sub_diag) {
        
    }




