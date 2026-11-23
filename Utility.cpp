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
        std::vector<double> sub_diagonal(size-1);
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
        sub_diagonal[size-2] = A[(size-2) * size + (size - 1)];
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

    EigenResult Utility::implicit_QR_shift(std::vector<double>& diag, std::vector<double>& sub_diag) {
        std::size_t size = diag.size()-1;
        double epsilon = std::numeric_limits<double>::epsilon();
        Matrix result = Matrix::identity(size+1);
        double* result_daten = result.data();
        std::size_t max_iter = 0;
        while(size > 0) {
            if(max_iter > 30 * diag.size() * 2)  throw std::runtime_error("QR-Iteration konvergiert nicht");
            max_iter++;

            for (std::size_t k = 0; k < size; ++k) {
                if (std::abs(sub_diag[k]) <= epsilon * (std::abs(diag[k]) + std::abs(diag[k+1])))
                    sub_diag[k] = 0.0;
            }

            if (sub_diag[size-1] == 0.0) { --size; continue; }

            std::size_t l = size - 1;
            while (l > 0 && sub_diag[l-1] != 0.0) --l;
            
            //erster schritt
            //eigenwert bestimmen
            double a = diag[size-1];
            double d = diag[size];
            double b = sub_diag[size-1];          // Index prüfen, siehe oben

            double delta = (a - d) / 2;
            double sgn = (delta >= 0.0) ? 1.0 : -1.0;
            double mu = d - sgn * b * b / (std::abs(delta) + std::hypot(delta, b));

            // Iteration über die Matrix mit bulge
            // bulge muss noch berechnet werden
            double bulge = 0.0, c, s, r, a_giv; // Bulge, c s r und a von givens
            //Erster schritt Bulge erstellen
            a_giv = diag[l]- mu;
            r = std::hypot(a_giv, sub_diag[l]);
            if(r == 0) {
                c = 1.0;
                s = 0.0;
            }
            else {
                c = a_giv / r;
                s = sub_diag[l] / r;
            }
            
            double cc = c * c;
            double ss = s * s;
            double cs = c * s;
            double d0 = diag[l];
            double d1 = diag[l+1];
            double e0 = sub_diag[l];

            diag[l] = cc * d0 + 2 * cs * e0 + ss * d1;
                    
            diag[l+1] = ss * d0 - 2 * cs * e0 + cc * d1;

            sub_diag[l] = (cc - ss) * e0 + cs * (d1 - d0);
            for(std::size_t i = 0; i <= size; ++i) {
                double x = result_daten[i];
                double y = result_daten[(size-1) + i];

                result_daten[i] = c * x + s * y;
                result_daten[(size-1) + i] = -s * x + c * y; 
            }

            if (size >= l + 2) {
                double e1 = c * sub_diag[l+1];
                bulge = s * sub_diag[l+1];
                sub_diag[l+1] = e1;
            }
            
            for(std::size_t i = l; i + 1 < size ; ++i) {
                a_giv = sub_diag[i];
                r = std::hypot(a_giv, bulge);
                if(r == 0) {
                    c = 1.0;
                    s = 0.0;
                }
                else {
                    c = a_giv / r;
                    s = bulge / r;
                }
                // Werte updaten, bulge muss nicht geupdated werden weil auf 0
                sub_diag[i] = r;

                // zu berechnen : subdiagonal element, zwei diagonalelemente und den bulge      
                cc = c * c;
                ss = s * s;
                cs = c * s;
                d1 = diag[i+1];
                double d2 = diag[i+2];
                double e1 = sub_diag[i+1];

                diag[i+1] = cc * d1 + 2 * cs * e1 + ss * d2;
                    
                diag[i+2] = ss * d1 - 2 * cs * e1 + cc * d2;

                sub_diag[i+1] = (cc - ss) * e1 + cs * (d2 - d1);
                
                if(i + 2 < size ) {
                    // nur wenn neue bulge entsteht
                    
                    double e2 = c * sub_diag[i+2];
                    
                    bulge = s * sub_diag[i+2]; 

                    sub_diag[i+2] = e2;
                }
                // Matrix update
                for(std::size_t j = 0; j <= size; ++j) {
                double x = result_daten[(size-1) * i + j];
                double y = result_daten[(size-1) * (i+1) + j];

                result_daten[(size-1) * i + j] = c * x + s * y;
                result_daten[(size-1) * (i+1) + j] = -s * x + c * y;   
                }
            }
        }

        std::sort(diag.begin(), diag.end(), std::greater<double>());
        return EigenResult{diag, result};
    }




