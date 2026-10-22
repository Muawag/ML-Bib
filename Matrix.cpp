#include "Matrix.h"
#include <stdexcept>
#include <string>
#include <cmath>
#include "SVD.h"
#include <random>
#include <cassert>

Matrix::Matrix(int rows, int cols) {
    size_.rows = rows;
    size_.cols = cols;
    scaled = false;
    //matrix_.resize(rows, std::vector<double>(cols, 0.0));
    matrix_as_vec.resize(rows * cols , 0.0);
}

Matrix::Matrix(const Matrix_Size& size) : size_(size) {
    scaled = false;
    //matrix_.resize(size.rows, std::vector<double>(size.cols, 0.0));
    matrix_as_vec.resize(size.rows * size.cols, 0.0);
}
Matrix::Matrix(const std::vector<std::vector<double>>& vecs) : scaled(false) {
    if(vecs.empty()) {
        size_.rows = 0;
        size_.cols = 0;
        scaled = false;
        return;
    }
    size_.rows = vecs.size();
    size_.cols = vecs.at(0).size();
    matrix_as_vec.reserve(size_.rows * size_.cols);
    for(std::size_t i = 0; i < size_.rows; ++i) {
        if(vecs[i].size() != size_.cols) throw std::runtime_error("Zeilen muessen alle gleich gross sein");
        matrix_as_vec.insert(matrix_as_vec.end(), vecs[i].begin(), vecs[i].end());
    }
}

Matrix::Matrix(const DataMatrix& d) : scaled(false) {
    if(!d.only_Numbers()) {
        throw std::runtime_error("Datamatrix darf nur numerische Einträge haben");
    }
    size_ = d.get_Size();
    if(size_.rows == 0) throw std::runtime_error("Datenmatrix war leer");
    matrix_as_vec.resize(size_.rows * size_.cols);
    const auto& raw = d.get_Raw();
    for(std::size_t i = 0; i < size_.rows; ++i) {
        for(std::size_t j = 0; j < size_.cols; ++j) {
            matrix_as_vec[i * size_.cols + j] = raw[j]->get_Element_as_Double(i);
        }
    }
}

Matrix::Matrix(int rows, int cols, double value) : scaled(false) {
    size_.rows = rows;
    size_.cols = cols;
    matrix_as_vec.resize(rows * cols , value);
}

std::span<double> Matrix::operator[](std::size_t row) {
    assert(row < size_.rows);
    return std::span<double>(matrix_as_vec.data() + row * size_.cols, size_.cols);
}

std::span<const double> Matrix::operator[](std::size_t row) const {
    assert(row < size_.rows);
    return std::span<const double>(matrix_as_vec.data() + row * size_.cols, size_.cols);
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
    m.matrix_as_vec = vec;
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
    for(std::size_t k = 0; k < this->matrix_as_vec.size(); ++k) {
       result.matrix_as_vec[k] = this->matrix_as_vec[k] + m2.matrix_as_vec[k]; 
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
        ret.push_back(matrix_as_vec[i * size_.cols + index]);
    }
    return ret;
}

Matrix Matrix::operator*(const Matrix& m2) const {
    if (size_.cols != m2.size_.rows) {
        throw std::runtime_error("Spaltenzahl der linken Matrix muss der Zeilenzahl der rechten entsprechen");
    }
    const std::size_t n = size_.rows;
    const std::size_t m = size_.cols;      
    const std::size_t p = m2.size_.cols;

    Matrix result(n, p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < m; ++k) {
            const double a = matrix_as_vec[i * m + k];
            for (std::size_t j = 0; j < p; ++j) {
                result.matrix_as_vec[i * p + j] += a * m2.matrix_as_vec[k * p + j];
            }
        }
    }
    return result;
}

void Matrix::print_Matrix() const {
    for(std::size_t i = 0; i < size_.rows; ++i) {
        for(std::size_t j = 0; j < size_.cols; ++j) {
            std::cout << matrix_as_vec[i * size_.cols + j] << "\t"; 
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::transpose() const {
    Matrix result(size_.cols, size_.rows);
    for(std::size_t i = 0; i < size_.rows; ++i) {
        for(std::size_t j = 0; j < size_.cols; ++j) {
            result.matrix_as_vec[j * result.size_.cols + i] = matrix_as_vec[i * size_.cols + j];
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
    const bool is_vec1 = (m1.size_.rows == 1 || m1.size_.cols == 1);
    const bool is_vec2 = (m2.size_.rows == 1 || m2.size_.cols == 1);
    if (!is_vec1 || !is_vec2) {
        throw std::runtime_error("Beide Matrizen müssen Vektoren sein");
    }
    if (m1.matrix_as_vec.size() != m2.matrix_as_vec.size()) {
        throw std::runtime_error("Vektoren müssen gleich lang sein");
    }

    double ret = 0.0;
    for (std::size_t i = 0; i < m1.matrix_as_vec.size(); ++i) {
        ret += m1.matrix_as_vec[i] * m2.matrix_as_vec[i];
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
            result.matrix_as_vec[i * result.size_.cols + j] = j < index ? matrix_as_vec[i * size_.cols + j] : matrix_as_vec[i * size_.cols + j +1];
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
            result.matrix_as_vec[i * result.size_.cols + j] = i < index ? matrix_as_vec[i * size_.cols + j] : matrix_as_vec[(i+1) * size_.cols + j];
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
            result.matrix_as_vec[i * result.size_.cols + j] = matrix_as_vec[(i-row_offset) * size_.cols + (j - col_offset)];
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
    const std::size_t n = diag.size();
    Matrix result(n,n);
    for(std::size_t i = 0; i < diag.size(); ++i) {
        result[i][i] = diag[i];
    }
    return result;
}
std::vector<double> Matrix::get_Diagonal_Vector() const {
    std::vector<double> diag;
    std::size_t max = size_.rows > size_.cols ? size_.cols : size_.rows;
    for(std::size_t i = 0; i < max; ++i) {
        diag.push_back(matrix_as_vec[i * size_.cols + i]);
    }
    return diag;
}

Matrix Matrix::make_Square() const {
    std::size_t size = size_.rows > size_.cols ? size_.cols : size_.rows;
    Matrix result = Matrix::identity(size);
    for(std::size_t i = 0; i < size; ++i) {
        for(std::size_t j = 0; j < size; ++j) {
            result[i][j] = matrix_as_vec[i * size_.cols + j];
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
            result[i][j] = matrix_as_vec[i * size_.cols + j];
        }
    }
    return result;
}

Matrix Matrix::resize(const Matrix_Size& size) const {
    Matrix result(size.rows, size.cols);
    for(std::size_t i = 0; i < size.rows; ++i) {
        for(std::size_t j = 0; j < size.cols; j++) {
            result[i][j] = matrix_as_vec[i * size_.cols + j];
        }
    }
    return result;
}

Matrix Matrix::get_Upper_Trig_From_Vec(const std::vector<std::vector<double>>& rows, const Matrix_Size& size) {
    Matrix result(size);
    for(std::size_t i = 0; i < rows.size(); ++i) {
        for(std::size_t j = 0; j < rows[i].size(); ++j) {
            std::size_t pos = result.size_.cols - rows[i].size() + j;
            result.matrix_as_vec[i * result.size_.cols + pos] = rows[i][j];
        }
    }
    return result;
}

bool Matrix::is_Vector() const {
    return(size_.cols == 1 || size_.rows == 1);
}

Matrix Matrix::random(const Matrix_Size& size, double mean, double stddev) {
    Matrix result(size);
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<double> dist(mean, stddev);
    for(std::size_t i = 0; i < result.size_.rows; ++i) {
        for(std::size_t j = 0 ; j < result.size_.cols; ++j) {
            result[i][j] = dist(gen);
        }
    }
    return result;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    for(int i = 0; i < size_.rows; i++)
        for(int j = 0; j < size_.cols; j++)
            matrix_as_vec[i * size_.cols + j] -= other.matrix_as_vec[i * other.size_.cols + j];
    return *this;
}

void Matrix::set_scaled(bool scaled) {
    this->scaled = scaled;
}

bool Matrix::is_scaled() const {
    return scaled;
}

Matrix Matrix::get_Matrix_With_Value(const Matrix_Size& size, double val) {
    return Matrix(size.rows, size.cols, val);
}
