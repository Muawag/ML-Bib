#pragma once
#include <numeric>
#include <iostream>
#include "Matrix_Size.h"
#include "NumberDataMatrix.h"
#include <vector>
#include "DataMatrix.h"
#include <span>
#include <random>
#include <chrono>
#include <immintrin.h>


class Matrix {
    public:
        Matrix();
        Matrix(int rows, int cols);
        Matrix(int rows, int cols, double value);
        Matrix(const Matrix_Size& size);
        Matrix(const std::vector<std::vector<double>>& vecs);
        Matrix(const DataMatrix& d);
        
        static Matrix identity(std::size_t size);
        static Matrix from_Vector(const std::vector<double>& vec, bool transpose);
        static Matrix get_Einheitsvektor(std::size_t size, std::size_t index, bool transpose);
        static Matrix get_Matrix_With_Value(const Matrix_Size& size, double val);
        static double inner_produkt(const Matrix& m1, const Matrix& m2);
        static Matrix get_Givens(std::size_t size, std::size_t i, double c, double s);
        static Matrix get_Diagonal(const std::vector<double>& diag);
        static Matrix random(const Matrix_Size& size, double lower, double upper);
        static Matrix get_Matrix_with_random(const Matrix_Size& size, std::mt19937& rnd, std::uniform_real_distribution<double>& dist);
        static Matrix get_Upper_Trig_From_Vec(const std::vector<std::vector<double>>& rows, const Matrix_Size& size);
        static Matrix old_mult(const Matrix& a, const Matrix& b);
        static double max_matrix_diff(const Matrix& a, const Matrix& b);


        Matrix operator+(const Matrix& m2) const;
        Matrix operator-(const Matrix& m2) const;
        Matrix operator*(const Matrix& m2) const; 
        Matrix& operator-=(const Matrix& other);
        std::span<double> operator[](std::size_t index);
        std::span<const double> operator[](std::size_t index) const;
        double& operator()(int i, int j);
        double operator()(int i, int j) const;
        template<typename T>
        requires std::is_arithmetic_v<T>
        Matrix operator*(T scalar) const {
            Matrix result(size_);
            for(std::size_t i = 0; i < size_.rows; ++i) {
                for(std::size_t j = 0; j < size_.cols; ++j) {
                    result[i][j] = (*this)[i][j] * static_cast<double>(scalar); 
                }
            }
            return result;
        }
        template<typename T>
        requires std::is_arithmetic_v<T>
        friend Matrix operator*(T scalar, const Matrix& m) {
            return m * scalar;
        }
        template<typename T>
        requires std::is_arithmetic_v<T>
        Matrix operator/(T scalar) {
            return *this * (1.0/ static_cast<double>(scalar));
        }
        
        Matrix_Size get_Size() const;
        void print_Matrix() const;    
        Matrix transpose() const;
        const std::vector<double> get_Column(std::size_t index) const;
        Matrix drop_Spalte(std::size_t index) const;
        Matrix drop_Zeile(std::size_t index) const;
        Matrix einbetten_u_r(std::size_t size) const;
        
        std::vector<double> get_Diagonal_Vector() const;
        Matrix make_Square() const;
        Matrix einbetten_o_l(std::size_t size) const;
        Matrix resize(const Matrix_Size& size) const;
        bool is_Vector() const;
        void set_scaled(bool scaled);
        bool is_scaled() const;
        double* data();
        const double* data() const;
        int ld() const;
        

    private:
        void initialize_Matrix_Vector();
        

    private:
        std::vector<double> matrix_as_vec;
        Matrix_Size size_;
        bool scaled;
};
