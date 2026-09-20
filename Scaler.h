#pragma once
#include <vector>
#include "Matrix.h"

class Scaler {
    public:
        virtual ~Scaler() = default;
        virtual void train(const Matrix& matrix) = 0;
        virtual Matrix train_apply(const Matrix& matrix) = 0;
        virtual Matrix apply(const Matrix& matrix) = 0;
    protected:
        bool trained;
};

class StandartScaler : public Scaler {
    public:
        StandartScaler();
        void train(const Matrix& matrix) override;
        Matrix train_apply(const Matrix& matrix) override;
        Matrix apply(const Matrix& matrix) override;
    private:
        double get_Mittelwert(const std::vector<double>& daten);
        double get_Standartabweichung(const std::vector<double>& daten, int index);
    private:
        std::vector<double> mittelwerte;
        std::vector<double> standartabweichungen;
};