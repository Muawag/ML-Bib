#pragma once
#include "SVMBinary.h"
#include <iostream>
#include <vector>
#include <map>
#include "Matrix.h"
#include "Classifier.h"
#include "DataMatrix.h"
#include "Utility.h"


template<typename T, typename Kernel>
class SVM : public Classifier<T> {
    public:
        SVM(const double C, const double tol, std::size_t max_passes, Kernel k) : C(C), tol(tol), 
            max_passes(max_passes), kernel_function(k) {
            }

        
        void train(const Matrix& x_daten, const DataMatrix& y_daten) override {
            auto[y_daten_irr, label_to_index, index_to_label, possible_classes] = Utility::y_DataMatrix_Converter(y_daten);
            for(std::size_t i = 0; i < possible_classes; ++i) {
                SVMBinary<T, Kernel> svm(C, tol, max_passes, kernel_function);
                svm.train_one_vs_rest(x_daten, y_daten, index_to_label.at(i));
                svm_vec.push_back(svm);
            }
        }

        DataMatrix predict(const Matrix& x_predict) const override {
            std::vector<Matrix> predicted_probs(x_predict.get_Size().rows);
            for(int i = 0; i < possible_classes; ++i) {
                predicted_probs[i] = svm_vec[i].get_predicted_Values(x_predict);
            }

            std::vector<T> predict_vec(x_predict.get_Size().rows);
            for(std::size_t i = 0; i < x_predict.get_Size().rows; ++i){
                double max_prob = 0.0;
                int index = 0;
                for(std::size_t j = 0; j < predicted_probs.size(); ++j) {
                    if(predicted_probs[j] >= max_prob) {
                        max_prob = predicted_probs[j];
                        index = j;
                    }
                }
                predict_vec[i] = index_to_label.at(index);
            }

            return DataMatrix({"Predicted Vals"}, std::move(predict_vec));
        }

        Matrix predict_proba(const Matrix& x_predict) const override {
            return Matrix::identity(2); // Nur Vollständigkeitshalber
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