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
    csv_matrix.print_Matrix();
    //csv_matrix.one_hot_encode("species", {"setosa"});
    //csv_matrix.print_Matrix();
    auto[train_daten, test_daten] = TrainTestSplit::split(csv_matrix, 0.8, "species");
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
    logReg.train(pca_x_train, predict_y, 2000, 0.2, 42);
    DataMatrix test_res_vec = logReg.predict(pca_x_test);

    std::cout << "Accuracy: " << Metrik::accuracy(predict_y_test, test_res_vec) << std::endl;
    /*
    DataMatrix cutouts = CSVLoader::load_CSV("cutouts.csv", false);
    auto[train_daten, test_daten] = TrainTestSplit::split(cutouts, 0.8, "3072");
    DataMatrix predict_y_train = train_daten.get_Spalte_as_Matrix_and_Drop("3072");
    DataMatrix predict_y_test = test_daten.get_Spalte_as_Matrix_and_Drop("3072");
    Matrix train_daten_x(train_daten);
    Matrix test_daten_x(test_daten);

    std::cout << "scaler" << std::endl;
    StandartScaler scaler;
    train_daten_x = scaler.train_apply(train_daten_x);
    test_daten_x = scaler.apply(test_daten_x);
    /*
    std::cout << "PCA" << std::endl;
    PCA pca;
    Matrix pca_train = pca.train_apply(train_daten_x);
    Matrix pca_test = pca.apply(test_daten_x);
    
    std::cout << "LogReg" << std::endl;
    LogisticRegression<std::string> logReg;
    logReg.train(train_daten_x, predict_y_train, 100000, 0.2, 100);
    DataMatrix predict_logreg = logReg.predict(test_daten_x);

    std::cout << "Accuracy: " << Metrik::accuracy(predict_y_test, predict_logreg) << std::endl;
    //predict_y_test.print_Matrix();
    //predict_logreg.print_Matrix();
    */
}