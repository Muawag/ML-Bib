#pragma once
#include <vector>
#include "Matrix.h"
#include "Regressor.h"
#include "Classifier.h"
#include "PCA.h"
#include "TrainTestSplit.h"
#include "DataMatrix.h"
#include <stdexcept>


std::vector<std::string> generate_header(const std::size_t);

using Metric_function = double(*)(const DataMatrix&, const DataMatrix&);



class CrossValidation {
    public :
        template<typename T>
        static DataMatrix test_for_hyperparameter(const std::size_t folds, Matrix& daten, DataMatrix& labels, std::unique_ptr<Classifier<T>>& model, const std::vector<double>& params, Metric_function function) {

            // primitive lsg erstmals
            std::size_t hyper_Params_count = params.size();

            std::vector<std::vector<double>> predictions(folds, std::vector<double>(hyper_Params_count, 0.0));

            

            for(std::size_t i = 0; i < folds; ++i) { // Folds schleife
                for(std::size_t j = 0; j < hyper_Params_count; ++j) {
                    //Model resetten

                    // trainieren nur temp parameter gerade
                    model.get()->train(daten, labels);
                    

                    // Predict mit dummy parametern
                    DataMatrix predict_y = model.get()->predict(daten);


                    // scores Speichern
                    predictions[i][j] = function(labels, predict_y);

                }
            }
        } 

};


std::vector<std::string> generate_Header(const std::size_t folds) {
    std::vector<std::string> header(folds);
    for(std::size_t i = 1; i <= folds; ++i) {
        header[i-1] = "Fold : " + std::to_string(i);
    }
} 