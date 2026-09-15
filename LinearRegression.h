#pragma once
#include "Regressor.h"


class LinearRegression : public Regressor {
    public:
        void train(const Matrix& x_daten, const Matrix& y_daten) override;
        Matrix predict(const Matrix& predict_daten) const override;
    private:
        std::vector<double> params;
};