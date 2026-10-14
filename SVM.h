#pragma once
#include "SVMBinary.h"
#include <iostream>
#include <vector>
#include <map>
#include "Matrix.h"
#include "Classifier.h"
#include "DataMatrix.h"
#include "Utility.h"
#include <limits>


template<typename T, typename Kernel>
class SVM : public Classifier<T> {
    public:
        SVM(const double C, const double tol, std::size_t max_passes, Kernel k) : C(C), tol(tol), 
            max_passes(max_passes), kernel_function(k) {}

        void train(const Matrix& x_daten, const DataMatrix& y_daten) override {
            svm_vec.clear();
            auto[y_daten_irr, l2i, i2l, classes] = Utility::y_DataMatrix_Converter<T>(y_daten);
            label_to_index = std::move(l2i);
            index_to_label = std::move(i2l);
            possible_classes = classes;
            for(std::size_t i = 0; i < possible_classes; ++i) {
                SVMBinary<T, Kernel> svm(C, tol, max_passes, kernel_function);
                svm.train_one_vs_rest(x_daten, y_daten, index_to_label.at(i));
                svm_vec.emplace_back(svm);
            }
        }

        DataMatrix predict(const Matrix& x_predict) const override {
            std::vector<Matrix> scores(possible_classes);
            for(int i = 0; i < possible_classes; ++i) {
                scores[i] = svm_vec[i].get_predicted_Values(x_predict);
            }

            std::vector<T> predict_vec(x_predict.get_Size().rows);
            for(std::size_t i = 0; i < x_predict.get_Size().rows; ++i){
                double max_prob = -std::numeric_limits<double>::infinity();
                int index = 0;
                for(std::size_t j = 0; j < possible_classes; ++j) {
                    if(scores[j][i][0] >= max_prob) {
                        max_prob = scores[j][i][0];
                        index = j;
                    }
                }
                predict_vec[i] = index_to_label.at(index);
            }
            return DataMatrix::make_single_col_vec("Predicted Values", std::move(predict_vec));
        }

        Matrix predict_proba(const Matrix& x_predict) const override {
            throw std::logic_error("SVMBinary::predict_proba ist nicht implementiert");
        }
        
    private:
        std::vector<SVMBinary<T, Kernel>> svm_vec;
        std::map<int, T> index_to_label;
        std::map<T, int> label_to_index;
        std::size_t possible_classes;
        double C;
        double tol;
        std::size_t max_passes;
        Kernel kernel_function;
};