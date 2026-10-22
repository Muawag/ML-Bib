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
#include <span>



template <typename T, typename Kernel>
class SVMBinary : public Classifier<T>{

    public:
        SVMBinary(const double C, const double tol, const std::size_t max_passes, Kernel k, T aim_type) : C(C), tol(tol), max_passes(max_passes),
        kernel_Funktion(std::move(k)), aim_type(aim_type) {}

        SVMBinary(const double C, const double tol, const std::size_t max_passes, Kernel k) : C(C), tol(tol),
        max_passes(max_passes), kernel_Funktion(std::move(k)) {}
        
        DataMatrix predict(const Matrix& predict_daten) const override{
            std::vector<T> result_vec(predict_daten.get_Size().rows);
            for(std::size_t i = 0; i < result_vec.size(); ++i) {
                result_vec[i] = (predict_single(predict_daten[i]) > 0.0) ? aim_type : other_type;
            }
            return DataMatrix::make_single_col_vec("Predict Values", std::move(result_vec));
        }

        Matrix get_predicted_Values(const Matrix& predict_daten) const {
            std::size_t n = predict_daten.get_Size().rows;
            Matrix result(n, 1);
            for(std::size_t i = 0; i < n; ++i) {
                result[i][0] = predict_single(predict_daten[i]);
            }
            return result;
        }

        void train(const Matrix& x_Daten, const DataMatrix& y_Daten) override{
            auto [y_Matrix, type_Key, class_Map, k] = Utility::y_DataMatrix_Converter<T>(y_Daten);
            y = Utility::y_DataMatrix_Converter_SVM<T>(y_Daten, aim_type);
            other_type = (class_Map.at(0) == aim_type) ? class_Map.at(1) : class_Map.at(0);
            label_to_int = type_Key;
            int_to_label = class_Map;
            possible_classes_count = k;
            if(k != 2) throw std::runtime_error("Muss binäre klassifikation sein");
            x = x_Daten;
            setup_Kernel_Matrix();
            a = Matrix::get_Matrix_With_Value(y.get_Size(), 0.0);
            b = 0.0;
            optimize();
        }

        void train_one_vs_rest(const Matrix& x_daten, const DataMatrix& y_daten, const T& positive) {
            y = Utility::y_DataMatrix_Converter_SVM<T>(y_daten, positive);
            x = x_daten;
            a = Matrix::get_Matrix_With_Value(y.get_Size(), 0.0);
            b = 0.0;
            setup_Kernel_Matrix();
            optimize();
        }

        Matrix predict_proba(const Matrix& predict_daten) const override {
            throw std::logic_error("SVMBinary::predict_proba ist nicht implementiert");
        }


    private:
        void optimize() {
            std::size_t passes = 0;
            std::size_t n = x.get_Size().rows;
            while(passes < max_passes) {
                std::size_t changed = 0;
                for(std::size_t i = 0; i < n; ++i) {
                    double E_i = error(i);

                    bool kkt_verletzt = (y[i][0] * E_i < -tol && a[i][0] < C) ||
                        (y[i][0] * E_i > tol && a[i][0] > 0);
                    
                    if(!kkt_verletzt) continue;

                    std::size_t j = std::rand() % (n - 1);
                    if(j >= i) ++j;

                    double E_j = error(j);

                    double a_i_alt = a[i][0];
                    double a_j_alt = a[j][0];

                    double eta = n_Val(i,j);
                    if(eta <= 0) continue;

                    a[j][0] = update_a_j(i,j, E_i, E_j);
                    set_Between_Boundaries(j,i);

                    if(std::abs(a[j][0] - a_j_alt) < 1e-5) {
                        a[j][0] = a_j_alt;
                        continue;
                    }

                    a[i][0] = update_a_i(i,j, a_j_alt);
                    update_b(E_i, E_j, a_i_alt, a_j_alt, i, j);
                    changed++;

                }
                passes = (changed > 0) ? 0 : passes + 1;
            }
        }
        double error(std::size_t index) const {
            double result = 0;
            return hyperebenen_Gleichung(index) - y[index][0];
        }

        double hyperebenen_Gleichung(std::size_t index) const {
            double sum = 0.0;
            for(std::size_t i = 0; i < x.get_Size().rows; ++i) {
                sum += (a[i][0] * y[i][0] * kernelMatrix[i][index]);
            }
            return sum + b;
        }

        double predict_single(std::span<const double> x_vec) const{
            double sum = b;
            for(std::size_t i = 0; i < x.get_Size().rows; ++i) {
                if(a[i][0] < 1e-8) continue;
                sum += a[i][0] * y[i][0] * kernel_Funktion(x[i], x_vec);
            }
            return sum;
        }

        double n_Val(std::size_t i, std::size_t j) {
            return kernelMatrix[i][i] + kernelMatrix[j][j] - (2 * kernelMatrix[i][j]); 
        }

        double update_a_j(std::size_t i, std::size_t j, double E_i, double E_j) {
            return a[j][0] + ((y[j][0] * (E_i - E_j)) / n_Val(i,j));
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
                if(i != r && i != s) sum += a[i][0] * y[i][0];
            }
            return -sum;
        }

        double help_func_Bound1(std::size_t r, std::size_t s) {
            return -(y[r][0] * ( y[s][0] * C - help_sum(r,s)));
        }

        double help_func_Bound2(std::size_t r, std::size_t s) {
            return y[r][0] * help_sum(r, s);
        }

        void update_b(double error_i, double error_j, double a_i_alt, double a_j_alt, std::size_t i, std::size_t j) {
            double b1 = b - error_i - y[i][0] * (a[i][0] - a_i_alt) * kernelMatrix[i][i] - 
                y[j][0] * (a[j][0] - a_j_alt) * kernelMatrix[i][j];

            double b2 = b - error_j - y[i][0] * (a[i][0] - a_i_alt) * kernelMatrix[i][j] - 
                y[j][0] * (a[j][0] - a_j_alt) * kernelMatrix[j][j];

            if(0 < a[i][0] && a[i][0] < C) b = b1;
            else if(0 < a[j][0] && a[j][0] < C) b = b2;
            else b = (b1 + b2) / 2;
        }

    private:
        Matrix a; // Lagrange Multiplier der support vektoren
        std::map<T, int> label_to_int;
        std::map<int, T> int_to_label; 
        double C; // Obere Schranke der as
        Matrix x; // Matrix aus allen support vektoren
        Matrix y; // Vektor aus den dazugehörigen labels
        double b = 0.0;
        int possible_classes_count;
        Matrix kernelMatrix;
        Kernel kernel_Funktion;
        double tol;
        std::size_t max_passes;
        T aim_type;
        T other_type;
};