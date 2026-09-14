#pragma once
#include <vector>
#include <string>
#include "Matrix_Size.h"
#include "DataMatrix.h"

class DataMatrix;

class NumberDataMatrix {
    public:
        NumberDataMatrix(const DataMatrix& data_matrix);
        NumberDataMatrix(int rows, int cols);
        void print_Matrix() const;
        Matrix_Size get_Size() const;
        const std::vector<double>& get_Column(int index) const;
        const bool is_scaled() const;
        void set_scaled(bool s);
        void set_Header(const std::vector<std::string>& header);
        const std::vector<std::string>& get_Header() const;
        std::vector<double>& operator[](std::size_t index);
        const std::vector<double>& operator[](std::size_t index) const;
        NumberDataMatrix operator+(const NumberDataMatrix& m1) const;
        NumberDataMatrix operator*(const NumberDataMatrix& m1) const;
        const std::vector<double> get_Row(int index) const;
        NumberDataMatrix transpose() const;
    private:
        std::vector<std::vector<double>> num_matrix_;
        std::vector<std::string> header_;
        Matrix_Size size_;
        bool scaled;
};

