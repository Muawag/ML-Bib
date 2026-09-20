#include "Scaler.h"
#include "NumberDataMatrix.h"
#include <iostream>
#include <cmath>


StandartScaler::StandartScaler() {
    trained = false;
}

double StandartScaler::get_Mittelwert(const std::vector<double>& daten) {
    if(daten.empty()) {
        std::cerr << "Keine Daten in einer Spalte" << std::endl;
        return 0.0;
    }
    double mean = 0;
    for(double point : daten) {
        mean += point;
    }
    return mean / daten.size();
}
double StandartScaler::get_Standartabweichung(const std::vector<double>& daten, int index) {
    if(daten.empty()) {
        std::cerr << "Keine Daten in einer Spalte" << std::endl;
        return 0.0;
    }
    double abw = 0;
    for(double point : daten) {
        double diff = point - mittelwerte[index];
        abw += diff * diff;
    }
    double abweichung = std::sqrtf(abw / daten.size());
    abweichung = abweichung == 0 ? 1 : abweichung;
    return abweichung;
}
void StandartScaler::train(const Matrix& matrix) {
    if(trained) {
        std::cout << "Überschreiben von alten Scaling Parametern" << std::endl;
    }
    mittelwerte.resize(matrix.get_Size().cols);
    standartabweichungen.resize(matrix.get_Size().cols);
    for(int i = 0; i < matrix.get_Size().cols ; ++i) {
        std::vector<double> spalte = matrix.get_Column(i);
        mittelwerte[i] = get_Mittelwert(spalte);
        standartabweichungen[i] = get_Standartabweichung(spalte, i);
    }
    trained = true;
}
Matrix StandartScaler::apply(const Matrix& matrix) {
    if(matrix.is_scaled()) {
        std::cerr << "Daten wurden schon skaliert" << std::endl;
        return Matrix{0,0};
    }
    if(!trained) {
        std::cerr << "Der Scaler wurde noch nicht trainiert" << std::endl;
        return Matrix{0,0};
    }
    Matrix scaled_matrix(matrix.get_Size().rows, matrix.get_Size().cols);
    //std::vector<std::string> header(matrix.get_Size().cols);
    //std::vector<std::string> header_old = matrix.get_Header();
    /*for(int i = 0; i < header.size(); ++i) {
        header[i] = header_old[i] + "_scaled";
    }
    scaled_matrix.set_Header(header);*/
    for(int i = 0; i < matrix.get_Size().rows; ++i) {
        for(int j = 0; j < matrix.get_Size().cols; ++j) {
            scaled_matrix[i][j] = (matrix[i][j] - mittelwerte[j]) / standartabweichungen[j];
        }
    }
    scaled_matrix.set_scaled(true);
    return scaled_matrix;
}

Matrix StandartScaler::train_apply(const Matrix& matrix) {
    train(matrix);
    return apply(matrix);
}