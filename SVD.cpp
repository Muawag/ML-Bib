#include "SVD.h"
#include "Matrix.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

double norm2(const std::vector<double>& vec);
void make_SingWert_pos(Matrix& U, Matrix& diag);
void cleanup_order(SVD_Matrizen& matricies);

Q_R_Matrizen SVD::q_r_step_spalte(Matrix& m){
    std::vector<double> a = m.get_Column(0);
    Matrix x = Matrix::from_Vector(a, false) + std::copysign(1.0, a[0]) * norm2(a) * Matrix::get_Einheitsvektor(a.size(), 0, false);
    Matrix qx = Matrix::identity(a.size()) - 2 * ((x * x.transpose()) / Matrix::inner_produkt(x,x));
    return {qx, qx * m};
}

Q_R_Matrizen SVD::q_r_step_zeile(Matrix& m) {
    std::vector<double> a = m[0];
    Matrix y = Matrix::from_Vector(a,false) + std::copysign(1.0, a[0]) * norm2(a) * Matrix::get_Einheitsvektor(a.size(), 0, false);
    Matrix qy = Matrix::identity(a.size()) - 2 * ((y * y.transpose()) / Matrix::inner_produkt(y,y));
    return {qy, m * qy};
} 

Matrix SVD::generate_Bidiagonalisierte_Matrix(const Matrix_Size& size, const std::vector<double>& as, const std::vector<double>& bs) {
    Matrix result(size);
    for(std::size_t i = 0; i < as.size(); ++i) {
        result[i][i] = as[i];
    }
    for(std::size_t i = 0; i < bs.size(); ++i) {
        result[i][i+1] = bs[i];
    }
    return result;
}

Bidiagonalisierte_Matrizen SVD::bidiagonalize(Matrix& m) {
    std::vector<Matrix> p_matricies;
    std::vector<Matrix> q_matricies;
    std::vector<double> as;
    std::vector<double> bs;
    Matrix_Size size = m.get_Size();
    bool square_Matrix = (size.rows == size.cols);
    if(size.rows >= size.cols) {
        int max = size.cols;
        for(int i = 0; i < max-1; ++i) {
            auto [p, m_neu] = q_r_step_spalte(m);
            p_matricies.push_back(p);
            as.push_back(m_neu[0][0]);
            m = m_neu.drop_Spalte(0);
            if(i == max-2) {
                bs.push_back(m[0][0]);
                m = m.drop_Zeile(0);
            }
            else{
                auto [q, m_neu2] = q_r_step_zeile(m);
                q_matricies.push_back(q);
                bs.push_back(m_neu2[0][0]);
                m = m_neu2.drop_Zeile(0);
            }
        }
        if(!square_Matrix) {
            auto [p, m_neu] = q_r_step_spalte(m);
            p_matricies.push_back(p);
            as.push_back(m_neu[0][0]);
        }
        else{
            as.push_back(m[0][0]);
        }
    }
    else{
        // Zweiter fall wird später behandelt
    }
    Matrix J = generate_Bidiagonalisierte_Matrix(size, as, bs);
    Matrix P = get_Matrix_From_Smaller(size.rows, p_matricies);
    Matrix Q = get_Matrix_From_Smaller(size.cols, q_matricies);
    return {P, J, Q};
}

double norm2(const std::vector<double>& vec) {
    return std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0));
}

Matrix SVD::get_Matrix_From_Smaller(std::size_t size, const std::vector<Matrix>& matricies) {
    Matrix result = Matrix::identity(size);
    for(std::size_t i = 0; i < matricies.size(); ++i) {
        result = result * matricies[i].einbetten_u_r(size);
    }
    return result;
}

SVD_Matrizen SVD::get_SVD_From_Bidiagonal(Bidiagonalisierte_Matrizen& matricies) {
    Matrix_Size original_size = matricies.J.get_Size();
    Matrix input = matricies.J.make_Square();
    Givens_Matrizen giv_zerl = givens_Iteration(input);
    giv_zerl.G_L = giv_zerl.G_L.einbetten_o_l(original_size.rows);
    Matrix U = matricies.P * giv_zerl.G_L.transpose();
    Matrix V = matricies.Q * giv_zerl.G_R;
    return {U, giv_zerl.D, V};
}

Givens_Matrizen SVD::givens_Iteration(Matrix& m) {
    Matrix_Size size = m.get_Size();
    Matrix U = Matrix::identity(size.rows);
    Matrix V = Matrix::identity(size.rows);
    if(size.rows != size.cols) {
        throw std::runtime_error("Matrix muss quadratisch sein");
    }
    int i = 0;
    while(true)  {
        for(std::size_t i = 0; i < size.rows - 1; ++i) {
            //double a_r = (i == 0) ? m[0][0] : m[i-1][i];
            //double b_r = (i == 0) ? m[0][1] : m[i-1][i+1];
            double a_r = m[i][i];
            double b_r = m[i][i+1];
            Matrix G_R = Matrix::get_Givens(size.rows, i, a_r, b_r).transpose();
            m = m * G_R;
            V = V * G_R;
            Matrix G_L = Matrix::get_Givens(size.rows, i, m[i][i], m[i+1][i]);
            m = G_L * m;
            U = G_L * U;
        }
        ++i;
        bool converged = true;
        for(std::size_t j = 0; j < size.rows-1; ++j) {
            if(std::abs(m[j][j+1]) > 1e-10) {
                converged = false;
                break;
            }
        }
        if(converged) break;
    }
    m = Matrix::get_Diagonal(m.get_Diagonal_Vector());
    return {V, m, U};
}

SVD_Matrizen SVD::svd(Matrix& m) {
    std::cout << "svd start" << std::endl;
    Matrix_Size size = m.get_Size();
    Bidiagonalisierte_Matrizen b = bidiagonalize(m);
    std::cout << "Bidiagonal fertid" << std::endl;
    SVD_Matrizen matricies = get_SVD_From_Bidiagonal(b);
    std::cout << "nur noch fertigstellen" << std::endl;
    return clean_up_Svd(size,matricies);
}

SVD_Matrizen SVD::clean_up_Svd(const Matrix_Size& original_size, SVD_Matrizen& matricies) {
    make_SingWert_pos(matricies.U, matricies.Sigma);
    cleanup_order(matricies);
    matricies.U = matricies.U.resize(original_size);
    return matricies;
}

void make_SingWert_pos(Matrix& U, Matrix& diag) {
    for(std::size_t i = 0; i < diag.get_Size().rows; ++i) {
        if(diag[i][i] < 0) {
            diag[i][i] = -diag[i][i];
            for(std::size_t j = 0; j < U.get_Size().rows; ++j) {
                U[j][i] = -U[j][i];
            }
        }
    }
}

void cleanup_order(SVD_Matrizen& matricies) {
    std::size_t n = matricies.Sigma.get_Size().rows;
    for(std::size_t i = 0; i < n - 1; ++i)  {
        for(std::size_t j = 0; j < n - i - 1; ++j) {
            if(matricies.Sigma[j][j] < matricies.Sigma[j+1][j+1]) {
                std::swap(matricies.Sigma[j][j], matricies.Sigma[j+1][j+1]);

                for(std::size_t k = 0; k < matricies.U.get_Size().rows; ++k) {
                    std::swap(matricies.U[k][j], matricies.U[k][j+1]);
                }

                for(std::size_t k = 0; k < matricies.V.get_Size().rows; ++k) {
                    std::swap(matricies.V[k][j], matricies.V[k][j+1]);
                }
            }
        }
    }
}

Q_R_Matrizen SVD::q_r_Decomposition(const Matrix& matrix) {
    std::size_t min_size = matrix.get_Size().cols < matrix.get_Size().rows ? matrix.get_Size().cols : matrix.get_Size().rows;
    Matrix m = matrix;
    std::vector<Matrix> q_Matricies;
    std::vector<std::vector<double>> rows;
    for(std::size_t i = 0; i < min_size; ++i) {
        if(m.get_Size().rows == 0 || m.get_Size().cols == 0) break;  
        auto[Q_n, m_neu] = SVD::q_r_step_spalte(m);
        q_Matricies.push_back(Q_n);
        rows.push_back(m_neu[0]);
        m = m_neu.drop_Zeile(0).drop_Spalte(0);
        Q_n.print_Matrix();
        m_neu.print_Matrix();
    }
    Matrix Q = get_Matrix_From_Smaller(matrix.get_Size().rows, q_Matricies);
    Matrix R = Matrix::get_Upper_Trig_From_Vec(rows, matrix.get_Size());
        
    return {Q,R};

}

 
