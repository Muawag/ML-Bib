#include "NumberDataMatrix.h"
#include <iostream>
#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <numeric>

void NumberDataMatrix::print_Matrix() const {
    for(int i = 0; i < header_.size(); ++i) {
        std::cout << header_.at(i) << "\t" ;
    }
    for(int i = 0; i < size_.rows; ++i) {
        std::cout << std::endl;
        for(int j = 0; j < size_.cols; ++j) {
            std::cout << num_matrix_.at(j).at(i) << "\t" ;
        }
    }
    std::cout << std::endl;
}
Matrix_Size NumberDataMatrix::get_Size() const {
    return size_;
}

NumberDataMatrix::NumberDataMatrix(const DataMatrix& data_matrix) {
    if(!data_matrix.only_Numbers()) {
        std::cerr << "DataMatrix darf nur numerische Spalten enthalten" << std::endl;
        return;
    }
    header_ = data_matrix.get_Header();
    num_matrix_.resize(data_matrix.get_Spaltencount());
    for(int i = 0; i < data_matrix.get_Spaltencount(); ++i) {
        num_matrix_[i] = data_matrix.get_Raw()[i].get()->get_Daten_as_Doubles();
    }
    size_.cols = data_matrix.get_Spaltencount();
    size_.rows = data_matrix.get_Eintraegecount();
    scaled = false;
}

NumberDataMatrix::NumberDataMatrix(int rows, int cols) {
    num_matrix_.resize(cols);
    for(auto& spalte : num_matrix_) {
        spalte.resize(rows);
    }
    size_.cols = cols;
    size_.rows = rows;
    scaled = false;
}
const std::vector<double>& NumberDataMatrix::get_Column(int index) const{
    return num_matrix_[index];
}

const bool NumberDataMatrix::is_scaled() const {
    return scaled;
} 

void NumberDataMatrix::set_Header(const std::vector<std::string>& header) {
    if(header.size() != size_.cols) {
        std::cerr << "Headeranzahl stimmt nicht mit der Spaltenanzahl überein" << std::endl;
        return;
    }
    header_ = header;
}

const std::vector<std::string>& NumberDataMatrix::get_Header() const {
    return header_;
}

std::vector<double>& NumberDataMatrix::operator[](std::size_t index) {
    if(index < size_.cols) return num_matrix_[index];
    std::cerr << "Index out of bounds" << std::endl;
    throw std::out_of_range("Index out of bounds");
}
const std::vector<double>& NumberDataMatrix::operator[](std::size_t index) const{
    if(index < size_.cols) return num_matrix_[index];
    std::cerr << "Index out of bounds" << std::endl;
    throw std::out_of_range("Index out of bounds");
}

void NumberDataMatrix::set_scaled(bool s) {
    scaled = s;
}

NumberDataMatrix NumberDataMatrix::operator+(const NumberDataMatrix& m1) const {
    if(this->size_ != m1.size_) {
        std::cerr << "Matrizen müssen für die Addition gleich groß sein" << std::endl;
        throw std::invalid_argument("Matrizen müssen für die Addition gleich groß sein");
    }
    NumberDataMatrix result(this->size_.rows, this->size_.cols);
    for(int i = 0; i < this->size_.cols; ++i) {
        for(int j = 0; j < this->size_.rows; ++j) {
            result[i][j] = (*this)[i][j] + m1[i][j];
        }
    }
    return result;
}

const std::vector<double> NumberDataMatrix::get_Row(int row) const {
    if(row >= this->size_.cols) {
        throw std::out_of_range("Index out of bounds");
    }
    std::vector<double> ret;
    for(int i = 0; i < this->size_.cols; ++i) {
        ret.push_back((*this)[i][row]);
    }
    return ret;
}

NumberDataMatrix NumberDataMatrix::operator*(const NumberDataMatrix& m1) const {
    if(this->size_.cols != m1.size_.rows) {
        std::cerr << "Matritzen haben nicht die passenden Grössen" << std::endl;
        throw std::invalid_argument("Matritzen haben nicht die passenden Grössen");
    }
    NumberDataMatrix result(this->size_.rows, m1.size_.cols);
    for(int i = 0; i < result.size_.rows; ++i) {
        std::vector<double> v1 = this->get_Row(i);
        for(int j = 0; j < result.size_.cols; ++j) {
            std::vector<double> v2 = m1.get_Column(j);
            result[j][i] = std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
        }
    }
    return result;
}

NumberDataMatrix NumberDataMatrix::transpose() const {
    NumberDataMatrix ret(this->size_.cols, this->size_.rows);
    for(int i = 0; i < this->size_.cols; ++i) {
        for(int j = 0; j < this->size_.rows; ++j) {
            ret[j][i] = (*this)[i][j];
        }
    }
    return ret;
}


