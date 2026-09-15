#include "Matrix.h"
#include <stdexcept>
#include <string>
#include <cmath>
#include "SVD.h"

Matrix::Matrix(int rows, int cols) {
    size_.rows = rows;
    size_.cols = cols;
    matrix_.resize(rows, std::vector<double>(cols, 0.0));
}

Matrix::Matrix(const Matrix_Size& size) : size_(size) {
    matrix_.resize(size.rows, std::vector<double>(size.cols, 0.0));
}
Matrix::Matrix(const std::vector<std::vector<double>>& vecs) {
    if(vecs.empty()) {
        size_.rows = 0;
        size_.cols = 0;
        return;
    }
    size_.rows = vecs.size();
    size_.cols = vecs.at(0).size();
    matrix_.resize(size_.rows);
    for(int i = 0; i <  size_.rows; ++i) {
        if(vecs[i].size() != size_.cols) {
            throw std::runtime_error("Alle Zeilen der Matrix muessen gleich gross sein");
        }
        matrix_[i] = vecs[i];
    }
}

Matrix::Matrix(const DataMatrix& d) {
    if(!d.only_Numbers()) {
        throw std::runtime_error("Datamatrix darf nur numerische Einträge haben");
    }
    size_ = d.get_Size();
    if(size_.rows == 0) {
        std::cerr << "Datamatrix war leer" << std::endl;
        return;
    }
    matrix_.resize(size_.rows);
    for(auto& row : matrix_) {
        row.resize(size_.cols);
    }
    for(std::size_t i = 0; i < size_.rows; ++i) {
        for(std::size_t j = 0; j < size_.cols; ++j) {
            matrix_[i][j] = d.get_Raw()[j]->get_Element_as_Double(i);
        }
    }
}

std::vector<double>& Matrix::operator[](std::size_t row) {
    return matrix_[row];
}

const std::vector<double>& Matrix::operator[](std::size_t row) const{
    return matrix_[row];
}

Matrix Matrix::identity(std::size_t n) {
    Matrix identity(n,n);
    for(int i = 0; i < n; ++i) {
        identity[i][i] = 1.0;
    }
    return identity;
}

Matrix Matrix::from_Vector(const std::vector<double>& vec, bool transpose) {
    Matrix m = transpose? Matrix(1, vec.size()) : Matrix(vec.size(), 1);
    if(!transpose) {
        for(std::size_t i = 0; i < m.size_.rows; ++i) {
            m.matrix_[i][0] = vec[i];
        }
    }
    else{
        m.matrix_[0] = vec;
    }
    return m;
}

Matrix_Size Matrix::get_Size() const{
    return size_;
}

Matrix Matrix::operator+(const Matrix& m2) const{
    if(this->size_ != m2.size_) {
        throw std::runtime_error("Matrizen muessen zum Addieren gleiche Groesse haben");
    }
    Matrix result(m2.size_.rows, m2.size_.cols);
    for(std::size_t i = 0; i < this->size_.rows; ++i) {
        for(std::size_t j = 0; j < this->size_.cols; ++j) {
            result.matrix_[i][j] = matrix_[i][j] + m2.matrix_[i][j];
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& m2) const {
    return *this + (m2 * -1);
}

const std::vector<double> Matrix::get_Column(std::size_t index) const {
    if(index >= size_.cols) {
        throw std::out_of_range("Invalide Spaltenangabe");
    }
    std::vector<double> ret;
    ret.reserve(size_.rows);
    for(std::size_t i = 0; i < size_.rows; ++i) {
        ret.push_back((*this)[i][index]);
    }
    return ret;
}

Matrix Matrix::operator*(const Matrix& m2) const {
    if(this->size_.cols != m2.size_.rows) {
        throw std::runtime_error("Matrizen muessen zum Multiplizieren gleich gross sein");
    }
    Matrix result(this->size_.rows, m2.size_.cols);
     for(std::size_t i = 0; i < this->size_.rows; ++i) {
        for(std::size_t k = 0; k < this->size_.cols; ++k) {
            for(std::size_t j = 0; j < m2.size_.cols; ++j) {
                result.matrix_[i][j] += this->matrix_[i][k] * m2.matrix_[k][j];
            }
        }
    }
    return result;
}

void Matrix::print_Matrix() const {
    for(std::size_t i = 0; i < size_.rows; ++i) {
        for(std::size_t j = 0; j < size_.cols; ++j) {
            std::cout << matrix_[i][j] << "\t"; 
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::transpose() const {
    Matrix result(size_.cols, size_.rows);
    for(std::size_t i = 0; i < size_.rows; ++i) {
        for(std::size_t j = 0; j < size_.cols; ++j) {
            result[j][i] = matrix_[i][j];
        }
    }
    return result;
}

Matrix Matrix::get_Einheitsvektor(std::size_t size, std::size_t index, bool transpose) {
    std::vector<double> vec(size, 0.0);
    vec[index] = 1;
    return from_Vector(vec,transpose);
}

double Matrix::inner_produkt(const Matrix& m1, const Matrix& m2) {
    size_t n1 = (m1.size_.rows == 1) ? m1.size_.cols : m1.size_.rows;
    size_t n2 = (m2.size_.rows == 1) ? m2.size_.cols : m2.size_.rows;
    
    if (n1 != n2)
        throw std::runtime_error("Vektoren muessen gleich gross sein");

    double ret = 0.0;
    for (size_t i = 0; i < n1; ++i) {
        double v1 = (m1.size_.rows == 1) ? m1.matrix_[0][i] : m1.matrix_[i][0];
        double v2 = (m2.size_.rows == 1) ? m2.matrix_[0][i] : m2.matrix_[i][0];
        ret += v1 * v2;
    }
    return ret;
}

Matrix Matrix::drop_Spalte(std::size_t index) const {
    if(index >= size_.cols) {
        throw std::out_of_range("Spaltenidex ist out of Range");
    }
    Matrix result(size_.rows, size_.cols-1);
    for(std::size_t i = 0; i < result.size_.rows; ++i) {
        for(std::size_t j = 0; j < result.size_.cols; ++j) {
            result[i][j] = j < index ? matrix_[i][j] : matrix_[i][j+1];
        }
    }
    return result;
}

Matrix Matrix::drop_Zeile(std::size_t index) const {
    if(index >= size_.rows) {
        throw std::out_of_range("Zeilenindex ist out of Range");
    }
    Matrix result(size_.rows-1, size_.cols);
    for(std::size_t i = 0; i < result.size_.rows; ++i) {
        for(std::size_t j = 0; j < result.size_.cols; ++j) {
            result[i][j] = i < index ? matrix_[i][j] : matrix_[i+1][j];
        }
    }
    return result;
}

Matrix Matrix::einbetten_u_r(std::size_t size) const {
    if(size < size_.cols || size < size_.rows) {
        throw std::runtime_error("Matrix ist zu gross um sie in die gewuenschte einheitsmatrix einzubetten");
    }
    Matrix result = identity(size);
    for(std::size_t i = size - size_.rows; i < size; ++i) {
        for(std::size_t j = size - size_.cols; j < size; ++j) {
            std::size_t row_offset = size - size_.rows;
            std::size_t col_offset = size - size_.cols;
            result[i][j] = matrix_[i-row_offset][j-col_offset];
        }
    }
    return result;
}

Matrix Matrix::get_Givens(std::size_t size, std::size_t i, double a, double b) {
    Matrix G = identity(size);
    double r = std::sqrt(a*a + b*b);
    double c = a/r;
    double s = b/r;
    G[i][i] = c;
    G[i+1][i+1] = c;
    G[i][i+1] = s;
    G[i+1][i] = -s;
    return G;
}

Matrix Matrix::get_Diagonal(const std::vector<double>& diag) {
    Matrix result = Matrix::identity(diag.size());
    for(std::size_t i = 0; i < diag.size(); ++i) {
        result[i][i] = diag[i];
    }
    return result;
}
std::vector<double> Matrix::get_Diagonal_Vector() const {
    std::vector<double> diag;
    std::size_t max = size_.rows > size_.cols ? size_.cols : size_.rows;
    for(std::size_t i = 0; i < max; ++i) {
        diag.push_back(matrix_[i][i]);
    }
    return diag;
}

Matrix Matrix::make_Square() const {
    std::size_t size = size_.rows > size_.cols ? size_.cols : size_.rows;
    Matrix result = Matrix::identity(size);
    for(std::size_t i = 0; i < size; ++i) {
        for(std::size_t j = 0; j < size; ++j) {
            result[i][j] = matrix_[i][j];
        } 
    }
    return result;
}

Matrix Matrix::einbetten_o_l(std::size_t size) const {
    Matrix result = identity(size);
    if(size < size_.cols || size < size_.rows) {
        throw std::runtime_error("Matrix ist zu gross um sie in die gewuenschte einheitsmatrix einzubetten");
    }
    for(std::size_t i = 0; i < size_.rows; ++i) {
        for(std::size_t j = 0; j < size_.cols; ++j) {
            result[i][j] = matrix_[i][j];
        }
    }
    return result;
}

Matrix Matrix::resize(const Matrix_Size& size) const {
    Matrix result(size.rows, size.cols);
    for(std::size_t i = 0; i < size.rows; ++i) {
        for(std::size_t j = 0; j < size.cols; j++) {
            result[i][j] = matrix_[i][j];
        }
    }
    return result;
}

Matrix Matrix::get_Upper_Trig_From_Vec(const std::vector<std::vector<double>>& rows, const Matrix_Size& size) {
    Matrix result(size);
    for(std::size_t i = 0; i < rows.size(); ++i) {
        for(std::size_t j = 0; j < rows[i].size(); ++j) {
            std::size_t pos = result.size_.cols - rows[i].size() + j;
            result.matrix_[i][pos] = rows[i][j];
        }
    }
    return result;
}