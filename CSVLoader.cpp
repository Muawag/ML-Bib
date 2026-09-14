#include "CSVLoader.h"
#include <iostream>
#include <stdexcept>
#include "Spalte.h"
#include <memory>

std::vector<std::vector<std::string>> get_CSV_as_Matrix(const std::string path);
bool is_Numeric(const std::string& s);
Column_Type check_Column_Type(const std::vector<std::string>& col);
bool is_Int(const std::string& s);
bool is_Bool(const std::string& s);
bool to_Bool(const std::string& s); 
std::unique_ptr<Spalte> make_Column(const std::vector<std::string>& col);

DataMatrix CSVLoader::load_CSV(const std::string path) {
    std::vector<std::vector<std::string>> matrix = get_CSV_as_Matrix(path);
    std::vector<std::string> header;
    for (size_t i = 0; i < matrix.size(); ++i) {
        header.push_back(matrix[i].at(0));
    }
    std::vector<std::unique_ptr<Spalte>> d_matrix;

    for (auto& col : matrix) {
        std::vector<std::string> ohne_header(col.begin() + 1, col.end());
        d_matrix.push_back(make_Column(ohne_header));
    }
    DataMatrix d(header, std::move(d_matrix));
    return d;

}
DataMatrix CSVLoader::load_CSV(const std::string path, const std::vector<std::string>& header) {
    std::vector<std::vector<std::string>> matrix = get_CSV_as_Matrix(path);
    std::vector<std::unique_ptr<Spalte>> d_matrix;
    for(auto& vec : matrix) {
        d_matrix.push_back(make_Column(vec));
    }
    DataMatrix d(header, std::move(d_matrix));
    return d;

}

std::vector<std::vector<std::string>> get_CSV_as_Matrix(const std::string path) {
    std::fstream file(path);
    if(!file.is_open()) {
        throw std::runtime_error("Pfad konnte nicht gefunden werden");
    }
    std::vector<std::vector<std::string>> matrix; 
    std::string line;
    int row = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        int col = 0;

        while (std::getline(ss, cell, ',')) {
            if (col >= matrix.size()) {
                matrix.resize(col + 1);
            }
            matrix[col].push_back(cell);
            col++;
        }
        row++;
    }
    return matrix;
}


bool is_Numeric(const std::string& s) {
    try {
        std::stod(s);
        return true;
    }
    catch (...) {
        return false;
    }
}

Column_Type check_Column_Type(const std::vector<std::string>& col) {
    Column_Type t = Column_Type::Bool;
    for(auto& s : col) {
        if(t == Column_Type::Bool && !is_Bool(s))
            t = Column_Type::Int;
        if(t == Column_Type::Int && !is_Int(s))
            t = Column_Type::Double;
        if(t == Column_Type::Double && !is_Numeric(s))
            t = Column_Type::String;
        if(t == Column_Type::String)
            break;
    }
    return t;
}

bool is_Int(const std::string& s) {
    try {
        std::size_t pos;
        std::stoi(s, &pos);
        return pos == s.size();
    }
    catch(...) {
        return false;
    }
}

bool is_Bool(const std::string& s) {
    return s == "True" || s == "true" || s == "False" || s == "false" || s == "1" || s == "0" ;
}

bool to_Bool(const std::string& s) {
    return (s == "True" || s == "true" || s == "1") ? true : false;
}

std::unique_ptr<Spalte> make_Column(const std::vector<std::string>& col) {
    switch(check_Column_Type(col)) {
        case Column_Type::Bool: {
            std::vector<bool> data;
            for(auto& s : col) data.push_back(to_Bool(s));
            return std::make_unique<TypedSpalte<bool>>(std::move(data));
        }
        case Column_Type::Int: {
            std::vector<int> data;
            for(auto& s : col) data.push_back(std::stoi(s));
            return std::make_unique<TypedSpalte<int>>(std::move(data));
        }
        case Column_Type::Double: {
            std::vector<double> data;
            for(auto& s : col) data.push_back(std::stod(s));
            return std::make_unique<TypedSpalte<double>>(std::move(data));
        }
        default:
            return std::make_unique<TypedSpalte<std::string>>(col);
    }
}

