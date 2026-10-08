#include <iostream>
#include <vector>
#include "Classifier.h"
#include "DataMatrix.h"
#include "Matrix.h"
#include "Matrix_Size.h"
#include <vector>
#include <map>
#include "Utility.h"



template <typename T>
class SVM : public Classifier<T>{

    SVM(const double C, const double tol, const std::size_t max_passes) : C(C), tol(tol), max_passes(max_passes) {

    }

    DataMatrix predict(const Matrix& predict_daten) const override{
        return nullptr;
    }

    void train(const Matrix& x_Daten, const DataMatrix& y_Daten) override{
        auto [y_Matrix, type_Key, class_Map, k] = Utility::y_DataMatrix_Converter<T>(y_Daten);
        y = y_Matrix;
        label_to_int = type_Key;
        int_to_label = class_Map;
        possible_classes_count = k;
        Matrix x_vecs = x_Daten;
    }


    private:
        void optimize()
        double error(std::size_t index) {

        }
        double hyperebenen_Gleichung(std::size_t index) {
            for()
        }


    private:
        Matrix a; // Lagrange Multiplier der support vektoren
        std::map<T, std::size_t> label_to_int;
        std::map<std::size_t, T> int_to_label; 
        double C; // Obere Schranke der as
        Matrix x; // Matrix aus allen support vektoren
        Matrix y; // Vektor aus den dazugehörigen labels
        double b = 0.0;
        int possible_classes_count;
};