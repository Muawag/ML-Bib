#include <iostream>
#include <vector>
#include "Matrix.h"
#include "Classifier.h"
#include <map>
#include <cmath>
#include <utility>

template <typename T>
class LogisticRegression : public Classifier {
    public:
        std::vector<T> predict(const Matrix& predict_daten) const override {
            std::vector<T> result;
            result.reserve(predict_daten.get_Size().rows);
            for(std::size_t i = 0; i < predict_daten.get_Size().rows; ++i) {
                result.push_back(predict_single(predict_daten[i]));
            }
            return result;
        }

        void train(const Matrix& x_daten, const Matrix& y_daten) override {

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

        double probability_Class(int class, const std::vector<double>& eintrag) const {
            auto [logits, nenner, max_logit] = logits_and_nenner(eintrag);
            return std::exp(logits[class] - max_logit) / nenner;
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

        void optimize(const Matrix& x_daten, const Matrix& y_daten, int epochen, double n) {
            weights = Matrix(possible_Classes_Count_, x_daten.get_Size().cols); // Default 0 inizialisierung
            for(int i = 0; i < epochen; ++i) {

            }
        }

        std::vector<double> gardient(const std::vector<double>& eintrag, int class) {
            double scalar = -probability_Class(class, eintrag);
            return - eintrag * scalar;
        }
        private:
            std::map<int, T> possible_Classes;
            std::map<T, int> possibility_Classes_Type_Key;
            int possible_Classes_Count_;
            Matrix weights;
            
};