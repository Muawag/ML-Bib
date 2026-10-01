#include "Spalte.h"
#include "DataMatrix.h"
#include "Scaler.h"
#include "Matrix.h"
#include "SVD.h"
#include "PCA.h"
#include "CSVLoader.h"
#include "LinearRegression.h"
#include "LogisticRegression.h"
#include "TrainTestSplit.h"
#include "Metrik.h"


int main() {
    DataMatrix csv_matrix = CSVLoader::load_CSV("iris.csv");
    auto[train_daten, test_daten] = TrainTestSplit::split(csv_matrix, 0.8);
    DataMatrix predict_y = train_daten.get_Spalte_as_Matrix_and_Drop("species");
    Matrix x_daten_train(train_daten);
    DataMatrix predict_y_test = test_daten.remove_Spalte("species");
    Matrix x_daten_test(test_daten);
    
    
    
    StandartScaler scaler;
    Matrix x_scaled_train = scaler.train_apply(x_daten_train);
    Matrix x_scaled_test = scaler.apply(x_daten_test);
    PCA pca;
    Matrix pca_x_train = pca.train_apply(x_scaled_train);
    Matrix pca_x_test = pca.apply(x_scaled_test);
   

    LogisticRegression<std::string> logReg;
    logReg.train(x_daten_train, predict_y, 2000, 0.1, 10);
    DataMatrix test_res_vec = logReg.predict(x_daten_test);

    std::cout << "Accuracy: " << Metrik::accuracy(predict_y_test, test_res_vec) << std::endl;
        
}