#include <iostream>
#include <vector>
#include "Matrix.h"
#include <string>

template <typename T>
class Classifier {
    public:
        virtual std::vector<T> predict(const Matrix& predict_daten) const = 0;
        virtual ~Classifier() = default;
        virtual void train(const Matrix& x_daten, const Matrix& y_daten) = 0;
        virtual Matrix predict_proba(const Matrix& predict_daten) const = 0;
};