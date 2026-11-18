#include "Matrix.h"
#include <stdexcept>
#include <string>
#include <cmath>
#include "SVD.h"
#include <random>
#include <cassert>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <algorithm>
#include <cassert>

void AddDot4x4(int, const double*, const double*, double*, int);

void AddDot8x4(int, const double*, const double*, double*, int);

void PackMatrixA( int, const double*, int, double*);

void PackMatrixB( int, const double*, int, double*);

void InnerKernel(int, int, int, const double*, int, const double*, double*, int);

void MY_MMult(int , int , int ,
              const double *, int ,
              const double *, int ,
              double *, int );

void Fast_MY_MMult(int, int, int, const double *, int, const double *, int, double *, int);

struct MatView {
    double *p;
    int ld;
    double& operator()(int i, int j) const {return p[j * ld + i];}
};

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
    /*if(m2.size_.cols % 4 != 0 ||size_.rows % 4 != 0) {
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
    }*/
    if (size_.cols != m2.size_.rows) {
        throw std::runtime_error("Spaltenzahl der linken Matrix muss der Zeilenzahl der rechten entsprechen");
    }
    const int m = static_cast<int>(size_.rows);      // Zeilen links
    const int k = static_cast<int>(size_.cols);      // gemeinsame Dimension
    const int n = static_cast<int>(m2.size_.cols);   // Spalten rechts

    Matrix C(Matrix_Size{m, n});                     // mit 0.0 initialisiert

    // Kernel rechnet column-major: C^T = B^T * A^T
    MY_MMult(n, m, k,
             m2.matrix_as_vec.data(), n,             // "A" des Kernels = B, ld = Spaltenzahl von B
             matrix_as_vec.data(), k,                // "B" des Kernels = A, ld = Spaltenzahl von A
             C.matrix_as_vec.data(), n);             // "C" des Kernels, ld = Spaltenzahl von C
    return C;
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

typedef union {
    __m128d v;
    double d[2];
} v2df_t;

void AddDot4x4(int k, const double *a, const double *b, double *c, int ldc) {

    int p;
    v2df_t
        c_00_c_10_vreg,  c_01_c_11_vreg,  c_02_c_12_vreg,  c_03_c_13_vreg,
        c_20_c_30_vreg,  c_21_c_31_vreg,  c_22_c_32_vreg,  c_23_c_33_vreg,
        a_0p_a_1p_vreg,
        a_2p_a_3p_vreg,
        b_p0_vreg,  b_p1_vreg,  b_p2_vreg,  b_p3_vreg;

        c_00_c_10_vreg.v = _mm_setzero_pd();
        c_01_c_11_vreg.v = _mm_setzero_pd();
        c_02_c_12_vreg.v = _mm_setzero_pd();
        c_03_c_13_vreg.v = _mm_setzero_pd();
        c_20_c_30_vreg.v = _mm_setzero_pd();
        c_21_c_31_vreg.v = _mm_setzero_pd();
        c_22_c_32_vreg.v = _mm_setzero_pd();
        c_23_c_33_vreg.v = _mm_setzero_pd();

        for( p = 0; p < k; p++) {
            a_0p_a_1p_vreg.v = _mm_load_pd( (double *) a);
            a_2p_a_3p_vreg.v = _mm_load_pd( (double *) ( a+2 ));
            a += 4;             

            b_p0_vreg.v = _mm_loaddup_pd( (double *) b);
            b_p1_vreg.v = _mm_loaddup_pd( (double *) (b+1));
            b_p2_vreg.v = _mm_loaddup_pd( (double *) (b+2));
            b_p3_vreg.v = _mm_loaddup_pd( (double *) (b+3));
            b += 4;

            c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
            c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
            c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
            c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;


            c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
            c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
            c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
            c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;

        }
        c[0] += c_00_c_10_vreg.d[0];  
        c[1 * ldc] += c_01_c_11_vreg.d[0];  
        c[2 * ldc] += c_02_c_12_vreg.d[0];  
        c[3 * ldc] += c_03_c_13_vreg.d[0]; 

        c[1] += c_00_c_10_vreg.d[1];  
        c[1 * ldc + 1] += c_01_c_11_vreg.d[1];  
        c[2 * ldc + 1] += c_02_c_12_vreg.d[1];  
        c[3 * ldc + 1] += c_03_c_13_vreg.d[1]; 

        c[2] += c_20_c_30_vreg.d[0];  
        c[1 * ldc + 2] += c_21_c_31_vreg.d[0];  
        c[2 * ldc + 2] += c_22_c_32_vreg.d[0];  
        c[3 * ldc + 2] += c_23_c_33_vreg.d[0]; 

        c[3] += c_20_c_30_vreg.d[1];
        c[1 * ldc + 3] += c_21_c_31_vreg.d[1];  
        c[2 * ldc + 3] += c_22_c_32_vreg.d[1];
        c[3 * ldc + 3] += c_23_c_33_vreg.d[1]; 


}

double* Matrix::data() {
    return matrix_as_vec.data();
}

const double* Matrix::data() const {
    return matrix_as_vec.data();
}

void PackMatrixA(int k, const double *a, int lda, double *a_to) {
    for (int j = 0; j < k; j++) {
        const double *col = &a[j * lda];
        for (int r = 0; r < 8; r++) *a_to++ = col[r];
    }
}

int Matrix::ld() const {
    return size_.rows;
}

double& Matrix::operator()(int i, int j) {
    return matrix_as_vec[ j * size_.rows + i];
}

double Matrix::operator()(int i, int j) const {
    return matrix_as_vec[ j * size_.rows + i];
}

void PackMatrixB(int k, const double *b, int ldb, double *b_to) {
	int i;
	const double
		*b_i0_pntr = &b[0 * ldb], *b_i1_pntr = &b[1 * ldb],
		*b_i2_pntr = &b[2 * ldb], *b_i3_pntr = &b[3 * ldb];

		for( i = 0; i < k; i++) {
			*b_to++ = *b_i0_pntr++;
			*b_to++ = *b_i1_pntr++;
			*b_to++ = *b_i2_pntr++;
			*b_to++ = *b_i3_pntr++;
		}
}

void InnerKernel(int m, int n, int k,
                 const double *a, int lda,
                 const double *packedB,
                 double *c, int ldc)
{
    std::vector<double> packedA(m * k);

    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 8) {
            if (j == 0)
                PackMatrixA(k, &a[i], lda, &packedA[i * k]);
            AddDot8x4(k, &packedA[i * k], &packedB[j * k],
                      &c[j * ldc + i], ldc);
        }
    }
}
constexpr int mc = 256;
constexpr int kc = 128;

void Fast_MY_MMult(int m, int n, int k,
              const double *a, int lda,
              const double *b, int ldb,
              double *c, int ldc)
{
    for (int p = 0; p < k; p += kc) {
    int pb = std::min(k - p, kc);
    std::vector<double> packedB(pb * n);
    for (int j = 0; j < n; j += 4)
        PackMatrixB(pb, &b[j * ldb + p], ldb, &packedB[j * pb]);

    for (int i = 0; i < m; i += mc) {
        int ib = std::min(m - i, mc);
        InnerKernel(ib, n, pb, &a[p * lda + i], lda,
                    packedB.data(), &c[i], ldc);
    }
}
}

void MY_MMult(int m, int n, int k, const double *a, int lda, const double *b, int ldb, double *c, int ldc) {
    int m8 = m - m%8;
    int n4 = n - n%4;
    
    if(m8 > 0 && n4 > 0) {
        Fast_MY_MMult(m8, n4, k, a, lda, b, ldb, c, ldc);
    }

        for (int j = n4; j < n; ++j) {
            for (int p = 0; p < k; ++p) {
                for (int i = 0; i < m; ++i) {
                    c[j*ldc + i] += a[p*lda + i] * b[j*ldb + p];
                }
            }
        }

    for (int j = 0; j < n4; ++j) {
        for (int p = 0; p < k; ++p) {
            for (int i = m8; i < m; ++i) {
                c[j*ldc + i] += a[p*lda + i] * b[j*ldb + p];
            }
        }
    }
}

Matrix Matrix::get_Matrix_with_random(const Matrix_Size& size, std::mt19937& rnd, std::uniform_real_distribution<double>& dist) {
    Matrix result(size);
    for (std::size_t i = 0; i < size.rows * size.cols; ++i) {
        result.matrix_as_vec[i] = dist(rnd);
    }
    return result;
}

Matrix Matrix::old_mult(const Matrix& a, const Matrix& b) {
        if (a.size_.cols != b.size_.rows) {
        throw std::runtime_error("Spaltenzahl der linken Matrix muss der Zeilenzahl der rechten entsprechen");
    }
    auto t0 = std::chrono::steady_clock::now();
    const std::size_t n = a.size_.rows;
    const std::size_t m = a.size_.cols;      
    const std::size_t p = b.size_.cols;

    Matrix result(n, p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < m; ++k) {
            const double a_num = a.matrix_as_vec[i * m + k];
            for (std::size_t j = 0; j < p; ++j) {
                result.matrix_as_vec[i * p + j] += a_num * b.matrix_as_vec[k * p + j];
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;   // Sekunden als double
    return result;
}

double Matrix::max_matrix_diff(const Matrix& a, const Matrix& b) {
    double maxdiff = 0.0;
    for (std::size_t i = 0; i < a.size_.rows * a.size_.cols; ++i)
        maxdiff = std::max(maxdiff, std::abs(a.matrix_as_vec[i] - b.matrix_as_vec[i]));
    return maxdiff;
}

void AddDot8x4(int k, const double *a, const double *b, double *c, int ldc) {
    __m256d c00 = _mm256_setzero_pd(), c01 = c00, c02 = c00, c03 = c00; // Zeilen 0-3
    __m256d c10 = c00, c11 = c00, c12 = c00, c13 = c00;                 // Zeilen 4-7

    for (int p = 0; p < k; ++p) {
        __m256d a0 = _mm256_loadu_pd(a);
        __m256d a1 = _mm256_loadu_pd(a + 4);

        __m256d b0 = _mm256_broadcast_sd(b + 0);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c10 = _mm256_fmadd_pd(a1, b0, c10);

        __m256d b1 = _mm256_broadcast_sd(b + 1);
        c01 = _mm256_fmadd_pd(a0, b1, c01);
        c11 = _mm256_fmadd_pd(a1, b1, c11);

        __m256d b2 = _mm256_broadcast_sd(b + 2);
        c02 = _mm256_fmadd_pd(a0, b2, c02);
        c12 = _mm256_fmadd_pd(a1, b2, c12);

        __m256d b3 = _mm256_broadcast_sd(b + 3);
        c03 = _mm256_fmadd_pd(a0, b3, c03);
        c13 = _mm256_fmadd_pd(a1, b3, c13);

        a += 8;
        b += 4;
    }

    double *c0 = c, *c1 = c + ldc, *c2 = c + 2*ldc, *c3 = c + 3*ldc;
    _mm256_storeu_pd(c0,     _mm256_add_pd(_mm256_loadu_pd(c0),     c00));
    _mm256_storeu_pd(c0 + 4, _mm256_add_pd(_mm256_loadu_pd(c0 + 4), c10));
    _mm256_storeu_pd(c1,     _mm256_add_pd(_mm256_loadu_pd(c1),     c01));
    _mm256_storeu_pd(c1 + 4, _mm256_add_pd(_mm256_loadu_pd(c1 + 4), c11));
    _mm256_storeu_pd(c2,     _mm256_add_pd(_mm256_loadu_pd(c2),     c02));
    _mm256_storeu_pd(c2 + 4, _mm256_add_pd(_mm256_loadu_pd(c2 + 4), c12));
    _mm256_storeu_pd(c3,     _mm256_add_pd(_mm256_loadu_pd(c3),     c03));
    _mm256_storeu_pd(c3 + 4, _mm256_add_pd(_mm256_loadu_pd(c3 + 4), c13));
}