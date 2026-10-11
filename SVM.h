#include <iostream>
#include <vector>
#include "Classifier.h"
#include "DataMatrix.h"
#include "Matrix.h"
#include "Matrix_Size.h"
#include <vector>
#include <map>
#include "Utility.h"
#include <functional>
#include <cmath>



template <typename T, typename Kernel>
class SVM : public Classifier<T>{

    public:
        SVM(const double C, const double tol, const std::size_t max_passes, Kernel k) : C(C), tol(tol), max_passes(max_passes),
        kernel_Funtion(std::move(k)) {

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
        void optimize() {

        }
        double error(std::size_t index) {
            double result = 0;
            return hyperebenen_Gleichung(index) - y[index][0];
        }

        double hyperebenen_Gleichung(std::size_t index) {
            double sum = 0.0;
            for(std::size_t i = 0; i < x.get_Size().rows; ++i) {
                sum += (a[i][0] * y[i][0] * kernelMatrix[i][index] + b) - y[index][0];
            }
            return sum;
        }

        double n_Val(std::size_t i, std::size_t j) {
            return kernelMatrix[i][i] + kernelMatrix[j][j] - (2 * kernelMatrix[i][j]); 
        }

        double update_a_j(std::size_t i, std::size_t j) {
            return a[j][0] + ((y[j][0] * (error(i) - error(j))) / n_Val(i,j));
        }

        double update_a_i(std::size_t i, std::size_t j, double a_j_alt) {
            return a[i][0] + y[i][0] * y[j][0] * (a_j_alt - a[j][0]);
        }

        void setup_Kernel_Matrix() {
            kernelMatrix = Matrix::identity(x.get_Size().rows);
            for(std::size_t i = 0; i < x.get_Size().rows; ++i) {
                for(std::size_t j = 0; j < x.get_Size().rows; ++j) {
                    kernelMatrix[i][j] = kernel_Funktion(x[i], x[j]);
                }
            }
        }

        void set_Between_Boundaries(std::size_t r, std::size_t s) {
            double lower;
            double upper;
            if(y[r][0] == y[s][0]) {
                lower = std::max(0.0, help_func_Bound1(r,s));
                upper = std::min(C, help_func_Bound2(r,s));
            }
            else {
                lower = std::max(0.0, help_func_Bound2(r,s));
                upper = std::min(C, help_func_Bound1(r,s));
            }
            if(a[r][0] > upper) a[r][0] = upper; 
            else if(a[r][0] < lower) a[r][0] = lower;
        }

        double help_sum(std::size_t r, std::size_t s) {
            double sum = 0;
            for(std::size_t i = 0 ; i < x.get_Size().rows; ++i) {
                if(i != r || i != s) sum += a[i][0] * y[i][0];
            }
            return -sum;
        }

        double help_func_Bound1(std::size_t r, std::size_t s) {
            return -(y[r][0] * ( y[s][0] * C - help_sum(r,s)));
        }

        double help_func_Bound2(std::size_t r, std::size_t s) {
            return y[r][0] * help_sum(r, s);
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
        Matrix kernelMatrix;
        Kernel kernel_Funtion;
        double to;
        int max_passes;
};