#pragma once
#include <iostream>
#include <vector>
#include "Matrix.h"
#include <string>
#include "DataMatrix.h"

template <typename T>
class Classifier {
    public:
        virtual std::vector<T> predict(const Matrix& predict_daten) const = 0;
        virtual ~Classifier() = default;
        virtual void train(const Matrix& x_daten, const DataMatrix& y_daten, int epochen, double lr, int batch_size) = 0; // muss noch schauen wies ist wekche matrizen übergeben werden sollten
        virtual Matrix predict_proba(const Matrix& predict_daten) const = 0;
};