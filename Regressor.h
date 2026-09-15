#pragma once
#include <iostream>
#include <string>
#include "Matrix.h" 
#include <vector>


class Regressor {
    public:
        virtual void train(const Matrix& x_daten, const Matrix& y_daten) = 0;
        virtual Matrix predict(const Matrix& predict_daten) const = 0;
        
        private:
};