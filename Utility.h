#pragma once
#include <random>
#include <iostream>
#include "Matrix.h"
#include <utility>
#include <algorithm>
#include <vector>
#include <tuple>
#include <map>
#include <stdexcept>

class Utility {
    public:
        static std::vector<std::pair<Matrix, Matrix>> get_Batches(const Matrix& x_vals, const Matrix& y_vals, int size);
        template<typename T>
        static std::tuple<Matrix, std::map<T, int>, std::map<int, T>, int> y_DataMatrix_Converter(const DataMatrix& y_daten) {
            Matrix result(y_daten.get_Size());
            std::map<T, int> result_map_type;
            std::map<int, T> result_map_class;
            int class_count = 0;
            TypedSpalte<T>* spalte = y_daten.get_Spalte<T>(0);
            for(std::size_t i = 0; i < spalte->size(); ++i) {
                if(!result_map_type.contains((*spalte)[i])) {
                    result_map_type.emplace((*spalte)[i], class_count);
                    result_map_class.emplace(class_count, (*spalte)[i]);
                    class_count++;
                }
                result[i][0] = result_map_type.at((*spalte)[i]);
            }
            return {result, result_map_type, result_map_class, class_count};
        }

        template<typename T> 
        static Matrix y_DataMatrix_Converter_SVM(const DataMatrix& y_daten, T specific_class) {
            Matrix result(y_daten.get_Size());
            bool specific_found = false;
            TypedSpalte<T>* spalte = y_daten.get_Spalte<T>(0);
            for(std::size_t i = 0 ; i < spalte->size(); ++i) {
                if((*spalte)[i] == specific_class) {
                    specific_found = true;
                    result[i][0] = 1;
                    continue;
                }
                result[i][0] = -1;
            }
            if(!specific_found) throw std::runtime_error("Übergebene Klasse war nicht vorhanden");
            return result;
        }


        static Matrix prepare_SVM_Matrix(const Matrix& labels, const int index); 
};