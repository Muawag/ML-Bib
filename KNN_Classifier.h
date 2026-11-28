#pragma once

#include "Matrix.h"
#include "Classifier.h"
#include <functional>
#include <algorithm>
#include <map> 



template<typename T>
class KNN_Classifier : public Classifier<T> {
    public:
        KNN_Classifier(const int k, std::function<double(const double*,const double*, const std::size_t)> metric): k(k), metric(metric) {}

        DataMatrix predict(const Matrix& predict_daten) const override {
            std::size_t x_cols = predict_daten.get_Size().cols;
            std::size_t x_rows = predict_daten.get_Size().rows;
            std::vector<T> result(x_rows);
            std::vector<std::tuple<double, T>> queue_;
            
            for(std::size_t i = 0; i < x_rows; ++i) {
                queue_.assign(k, {std::numeric_limits<double>::infinity(), T{}});
                const double* daten_ptr = daten_.data();
                const double* mat_ptr = predict_daten.data() + i * x_cols;
                
                for(std::size_t j = 0; j < daten_.get_Size().rows; ++j) {
                    double dist = metric(mat_ptr, daten_ptr, x_cols);
                    daten_ptr += x_cols;
                    append_queue(dist, labels[j], queue_);
                }
                result[i] = get_correct_type(queue_);
            }

            TypedSpalte<T> temp_Spalte(std::move(result));
            std::vector<std::unique_ptr<Spalte>> spalte;
            spalte.push_back(std::make_unique<TypedSpalte<T>>(std::move(temp_Spalte)));
            
            return DataMatrix({"Predicted"}, std::move(spalte));
        }


        void train(const Matrix& x_daten, const DataMatrix& y_daten) override {
            if(y_daten.get_Spaltencount() != 1) throw std::runtime_error("Labels dürfen nur eine spalte haben");
            labels = y_daten.get_Spalte<T>(0)->get_Raw_Vec();
            //labels.resize(y_daten.get_Eintraegecount());
            //std::copy(ptr, ptr + y_daten.get_Eintraegecount(), labels.begin());
            daten_ = x_daten; // Copy vielleicht später sinnvoller zu moven wg speicher//
        }
        
        Matrix predict_proba(const Matrix& predict_daten) const override {
            throw std::logic_error("Noch nicht implementiert");
        }
    
    private:
        static void append_queue(double dist, T label, std::vector<std::tuple<double,T>>& queue_) {
            if(dist >= std::get<0>(queue_.back())) return;
            auto pos = std::upper_bound(queue_.begin(), queue_.end(), dist,
                [](double d, const auto& elem) { return d < std::get<0>(elem); });

            std::move_backward(pos, queue_.end() - 1, queue_.end());
            *pos = {dist, std::move(label)};
        }
        T get_correct_type(std::vector<std::tuple<double,T>>& queue_) const {
            std::map<T, std::size_t> votes;
            T best{};
            std::size_t best_count = 0;

            for (const auto& [dist, label] : queue_) {
                if (dist == std::numeric_limits<double>::infinity()) break; 

                const std::size_t count = ++votes[label];
                if (count > best_count) {
                    best_count = count;
                    best = label;
                }
            }
            return best;
        }

    private:
        Matrix daten_;
        std::vector<T> labels;
        std::function<double(const double*, const double*, const std::size_t)> metric;
        int k;
    
};
