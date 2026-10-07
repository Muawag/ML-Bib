#include <iostream>
#include <vector>
#include "Classifier.h"
#include "DataMatrix.h"
#include "Matrix.h"
#include "Matrix_Size.h"
#include <vector>
#include <map>



template <typename T>
class SVM : public Classifier<T>{

    SVM(const double C, const double tol, const std::size_t max_passes) : C(C), tol(tol), max_passes(max_passes) {

    }

    DataMatrix predict(const Matrix& predict_daten) const override{
        return nullptr;
    }

    void train(const Matrix& x_Daten, const DataMatrix& y_Daten) override{
        
    }



    private:
        Matrix a; // Lagrange Multiplier der support vektoren
        std::map<T, std::size_t> label_to_int;
        double C; // Obere Schranke der as
        Matrix x; // Matrix aus allen support vektoren
        Matrix y; // Vektor aus den dazugehörigen labels
        double b = 0.0;
};