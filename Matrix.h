#pragma once
#include <numeric>
#include <iostream>
#include "Matrix_Size.h"
#include "NumberDataMatrix.h"
#include <vector>
#include "DataMatrix.h"


class Matrix {
    public:
        Matrix() : size_({0, 0}), matrix_(0) {}
        Matrix(int rows, int cols);
        Matrix(const Matrix_Size& size);
        //Matrix(const NumberDataMatrix& m); <-- Später implementieren
        Matrix(const std::vector<std::vector<double>>& vecs);
        Matrix(const DataMatrix& d); 
        static Matrix identity(std::size_t size);
        static Matrix from_Vector(const std::vector<double>& vec, bool transpose);
        static Matrix get_Einheitsvektor(std::size_t size, std::size_t index, bool transpose);
        Matrix operator+(const Matrix& m2) const;
        Matrix operator-(const Matrix& m2) const;
        Matrix operator*(const Matrix& m2) const;
        Matrix_Size get_Size() const;
        std::vector<double>& operator[](std::size_t index);
        const std::vector<double>& operator[](std::size_t index) const;
        void print_Matrix() const;

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

        Matrix transpose() const;
        const std::vector<double> get_Column(std::size_t index) const;
        static double inner_produkt(const Matrix& m1, const Matrix& m2);
        Matrix drop_Spalte(std::size_t index) const;
        Matrix drop_Zeile(std::size_t index) const;
        Matrix einbetten_u_r(std::size_t size) const;
        static Matrix get_Givens(std::size_t size, std::size_t i, double c, double s);
        static Matrix get_Diagonal(const std::vector<double>& diag);
        std::vector<double> get_Diagonal_Vector() const;
        Matrix make_Square() const;
        Matrix einbetten_o_l(std::size_t size) const;
        Matrix resize(const Matrix_Size& size) const;
        static Matrix get_Upper_Trig_From_Vec(const std::vector<std::vector<double>>& rows, const Matrix_Size& size);
    private:
        std::vector<std::vector<double>> matrix_;
        Matrix_Size size_;
        //bool is_Diagonal_;
        //bool is_Orthogonal_;
};
