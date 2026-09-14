#pragma once
#include "NumberDataMatrix.h"
#include <vector>

class Scaler {
    public:
        virtual ~Scaler() = default;
        virtual void train(const NumberDataMatrix& matrix) = 0;
        virtual NumberDataMatrix train_apply(const NumberDataMatrix& matrix) = 0;
        virtual NumberDataMatrix apply(const NumberDataMatrix& matrix) = 0;
    protected:
        bool trained;
};

class StandartScaler : public Scaler {
    public:
        StandartScaler();
        void train(const NumberDataMatrix& matrix) override;
        NumberDataMatrix train_apply(const NumberDataMatrix& matrix) override;
        NumberDataMatrix apply(const NumberDataMatrix& matrix) override;
    private:
        double get_Mittelwert(const std::vector<double>& daten);
        double get_Standartabweichung(const std::vector<double>& daten, int index);
    private:
        std::vector<double> mittelwerte;
        std::vector<double> standartabweichungen;
};