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
        std::size_t size = diag.size()-1;
        double epsilon = std::numeric_limits<double>::epsilon();
        while(size > 0) {// abbruchbedingung fehlt noch
            //erster schritt
            //eigenwert bestimmen
            double a = diag[size-1];
            double d = diag[size];
            double b = sub_diag[size-1];          // Index prüfen, siehe oben

            double mu = d;
            if (std::abs(b) > epsilon) {
                double delta = (a - d) / 2;
                double sgn = (delta >= 0.0) ? 1.0 : -1.0;
                mu = d - sgn * b * b / (std::abs(delta) + std::hypot(delta, b));
            }
            else size--;

            // Iteration über die Matrix mit bulge
            // bulge muss noch berechnet werden
            double bulge; // Bulge
            for(std::size_t i = 0; i < size - 1; ++i) {
                double a = sub_diag[i];
                double r = std::sqrt(a * a + bulge * bulge);
                // falls r ungleich 0
                double c = a / r;
                double s = bulge / r;
                // Werte updaten, bulge muss nicht geupdated werden weil auf 0
                sub_diag[i] = r;
                if(i + 2 != size ) {
                    double zwi1, zwi2;

                    zwi1 = c * diag[i+1] + s * sub_diag[i+1];
                    zwi2 = - s * diag[i+1] + c * sub_diag[i+1];
                    diag[i+1] = zwi1;

                    diag[i+2] = - s * sub_diag[i+1] + c * diag[i+2];

                    bulge = s * sub_diag[i+2];
                    sub_diag[i+2] = c * sub_diag[i+2];
                }
            }
        }
    }




