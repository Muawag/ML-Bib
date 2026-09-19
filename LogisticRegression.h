#pragma once
#include <iostream>
#include <vector>
#include "Matrix.h"
#include "Classifier.h"
#include <map>
#include <cmath>
#include <utility>
#include <random>
#include "Utility.h"

template <typename T>
class LogisticRegression : public Classifier<T> {
    public:
        std::vector<T> predict(const Matrix& predict_daten) const override {
            std::vector<T> result;
            result.reserve(predict_daten.get_Size().rows);
            for(std::size_t i = 0; i < predict_daten.get_Size().rows; ++i) {
                result.push_back(predict_single(predict_daten[i]));
            }
            return result;
        }

        void train(const Matrix& x_daten, const DataMatrix& y_daten, int epochen = 200, double lr = 0.05, int batch_size = 32) override {
            auto [y_matrix, type_key, class_map, k] = Utility::y_DataMatrix_Converter<T>(y_daten);
            possible_Classes_Count_ = k;
            possible_Classes = class_map;
            possibility_Classes_Type_Key = type_key;
            optimize(x_daten, y_matrix, epochen, lr, batch_size);
        }

        Matrix predict_proba(const Matrix& predict_daten) const override {
            Matrix result(predict_daten.get_Size().rows, possible_Classes_Count_);
            for (std::size_t i = 0; i < predict_daten.get_Size().rows; ++i) {
                auto [logits, nenner, max_logit] = logits_and_nenner(predict_daten[i]);
                for (std::size_t j = 0; j < possible_Classes_Count_; ++j){
                    result[i][j] = std::exp(logits[j] - max_logit) / nenner;
                }    
            }
            return result;
        }
    
        private:

        T predict_single(const std::vector<double>& eintrag) const {
            std::size_t best = 0;
            double best_logit = std::inner_product(weights[0].begin(), weights[0].end(), eintrag.begin(), 0.0);

            for (std::size_t i = 1; i < possible_Classes_Count_; ++i) {
                double logit = std::inner_product(weights[i].begin(), weights[i].end(), eintrag.begin(), 0.0);
                if (logit > best_logit) {
                    best_logit = logit;
                    best = i;
                }
            }
            return possible_Classes.at(best);
        }

        double probability_Class(int klasse, const std::vector<double>& eintrag) const {
            auto [logits, nenner, max_logit] = logits_and_nenner(eintrag);
            return std::exp(logits[klasse] - max_logit) / nenner;
        }

        std::tuple<std::vector<double>, double, double> logits_and_nenner(const std::vector<double>& eintrag) const {
            std::vector<double> logits(possible_Classes_Count_);
            for (std::size_t i = 0; i < possible_Classes_Count_; ++i) {
                logits[i] = std::inner_product(weights[i].begin(), weights[i].end(), eintrag.begin(), 0.0);
            }
            double max_logit = *std::max_element(logits.begin(), logits.end());
            double nenner = 0.0;
            for (double l : logits) {
                nenner += std::exp(l - max_logit);
            }
            return {logits, nenner, max_logit};
        }

        double negative_log_likleyhood(const Matrix& x_daten, const Matrix& y_daten) const {
            double sum = 0.0;
            for(std::size_t i = 0; i < x_daten.get_Size().rows; ++i) {
                sum += std::log(probability_Class(possibility_Classes_Type_Key.at(y_daten[i], x_daten[i])));
            }
            return -sum;
        }

        void optimize(const Matrix& x_daten, const Matrix& y_daten, int epochen, double n, int batch_size) {
            weights = Matrix::random({possible_Classes_Count_, x_daten.get_Size().cols}, 0.0, 0.01); 
            int batch_Size = batch_size;
            for(int e = 0; e < epochen; ++e) {
                auto batches = Utility::get_Batches(x_daten, y_daten, batch_Size);
                for(auto& [x_batch, y_batch] : batches) {
                    Matrix grad = gardient(x_batch, y_batch);
                    weights -= n * grad;
                }
            }
        }

        Matrix gardient(const Matrix& x_Batch, const Matrix& y_Batch) {
            Matrix P_T = predict_proba(x_Batch);
            Matrix Y = get_Y_one_Hot_Encoded(y_Batch);
            Matrix Diff = Y - P_T;

            return (Diff.transpose() * x_Batch) * -1.0;

        }
        private:
            Matrix get_Y_one_Hot_Encoded(const Matrix& y_Batch) const {
                Matrix Y(y_Batch.get_Size().rows, possible_Classes_Count_);
                for(std::size_t i = 0; i < y_Batch.get_Size().rows; ++i) {
                    Y[i][y_Batch[i][0]] = 1;
                }
                return Y;
            }

            void setup_Maps(const Matrix& y_Daten) {
                for(std::size_t i = 0; i < y_Daten.get_Size().rows; ++i) {
                    if(!possibility_Classes_Type_Key.contains(y_Daten[i][0])) {
                        possibility_Classes_Type_Key.emplace({y_Daten[i][0], possible_Classes_Count_});
                        possible_Classes.emplace({possible_Classes_Count_, y_Daten[i][0]});
                        possible_Classes_Count_++;
                    }
                }
            }
        private:
            std::map<int, T> possible_Classes;
            std::map<T, int> possibility_Classes_Type_Key;
            int possible_Classes_Count_;
            Matrix weights;
            
};
