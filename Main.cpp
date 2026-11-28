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
#include "SVM.h"
#include "ActivationFunktion.h"
#include "Kernel.h"
#include "PCAFast.h"
#include <chrono>
#include "ImageLoader.h"
#include <print>
#include "KNN_Classifier.h"

int main() {
    
    DataMatrix csv_matrix = CSVLoader::load_CSV("iris.csv");
    //csv_matrix.print_Matrix();
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
    PCAFast pca;
    Matrix pca_x_train = pca.train_apply(x_scaled_train);
    Matrix pca_x_test = pca.apply(x_scaled_test);
    //pca_x_train.print_Matrix();
   

    LogisticRegression<std::string> logReg(2000, 0.2, 42);
    logReg.train(pca_x_train, predict_y);
    DataMatrix test_res_vec_log = logReg.predict(pca_x_test);

    std::cout << "Accuracy LogReg: " << Metrik::accuracy(predict_y_test, test_res_vec_log) << std::endl;

    auto euclidean = [](const double* x1, const double* x2, std::size_t cols) {
        double dist = 0.0;
        for(std::size_t i = 0; i < cols; ++i) {
            dist += (x1[i] - x2[i]) * (x1[i] - x2[i]); 
        }
        return std::sqrt(dist);
    };

    KNN_Classifier<std::string> knn(5, euclidean);

    knn.train(pca_x_train, predict_y);
    DataMatrix test_vec_knn = knn.predict(pca_x_test);

    std::cout << "Accuracy KNN: " << Metrik::accuracy(predict_y_test, test_vec_knn) << std::endl;
    
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
    
    std::cout << "PCA" << std::endl;
    PCAFast pca;
    Matrix pca_train = pca.train_apply(train_daten_x);
    Matrix pca_test = pca.apply(test_daten_x);
    
    std::cout << "LogReg" << std::endl;
    LogisticRegression<std::string> logReg(2000, 0.2, 100);
    logReg.train(train_daten_x, predict_y_train);
    DataMatrix predict_logreg = logReg.predict(test_daten_x);

    std::cout << "Accuracy: " << Metrik::accuracy(predict_y_test, predict_logreg) << std::endl;
    
    std::cout << "SVM" << std::endl;
    SVM<std::string, RBFKernel> svm(1.0, 1e-3, 10, RBFKernel());
    svm.train(train_daten_x, predict_y_train);
    DataMatrix test_res_vec_svm = svm.predict(test_daten_x);

    std::cout << "Accuracy SVM: " << Metrik::accuracy(predict_y_test, test_res_vec_svm) << std::endl;
    //predict_y_test.print_Matrix();
    //predict_logreg.print_Matrix();
    
    /*
    SVM<std::string, RBFKernel> svm(1.0, 1e-3, 10, RBFKernel());
    svm.train(pca_x_train, predict_y);
    DataMatrix test_res_vec_svm = svm.predict(pca_x_test);

    std::cout << "Accuracy SVM: " << Metrik::accuracy(predict_y_test, test_res_vec_svm) << std::endl;


    
    std::vector<std::vector<double>> a1 = {
    { 1,  2,  0, -1,  3,  4,  1, -2},
    { 0,  3, -2,  5,  1,  0,  2,  1},
    { 4, -1,  1,  2, -3,  2,  0,  6},
    { 2,  0,  5, -2,  1, -1,  3,  0},
    {-3,  1,  2,  0,  4,  3, -1,  2}
    };
    Matrix testMat(a1);
    testMat = testMat.transpose();

    testMat.print_Matrix();
    PCA pca;
    Matrix pcaMatr = pca.train_apply(testMat);
    //pcaMatr.print_Matrix();
    
std::mt19937 rng(42);
std::uniform_real_distribution<double> dist(-1.0, 1.0);

Matrix A = Matrix::get_Matrix_with_random(Matrix_Size{800, 400000}, rng, dist);
//Matrix B = Matrix::get_Matrix_with_random(Matrix_Size{128000, 1000}, rng, dist);

StandartScaler s;
Matrix scaled = s.train_apply(A);

auto t0 = std::chrono::steady_clock::now();
PCAFast pca;
Matrix pcaMat = pca.train_apply(scaled);
auto t1 = std::chrono::steady_clock::now();

double sec = std::chrono::duration<double>(t1 - t0).count();
std::cout << "Fertig: " << sec << " Sekunden" << std::endl;
//Matrix c_ref = Matrix::old_mult(matA, matB);

//std::cout << "Diff: " << Matrix::max_matrix_diff(matC, c_ref) << std::endl;



//DataMatrix daten = CSVLoader::load_CSV("cutouts.csv", false);
//daten.remove_Spalte("3072");
//DataMatrix daten = CSVLoader::load_CSV("iris.csv");
//daten.remove_Spalte("species");
//Matrix matrix(daten);
//StandartScaler s;
//std::cout << matrix.get_Size() << std::endl;
//Matrix scaled = s.train_apply(matrix);
//PCAFast pca;
//Matrix pcaMat = pca.train_apply(scaled);
//PCA pca;
//Matrix pcaMat = pca.train_apply(scaled);
//pcaMat.print_Matrix();
//std::cout << pcaMat.get_Size() << std::endl;

/*std::mt19937 rng(42);
std::uniform_real_distribution<double> dist(-1.0, 1.0);

Matrix m = Matrix::get_Matrix_with_random(Matrix_Size{1000, 400000}, rng, dist);
StandartScaler s;
m = s.train_apply(m);
auto[a,b,c] = SVD::svd_gramm(m, 0.95);
std::cout << b.get_Size() << std::endl;
//std::cout << "Fertig, Dim: " << b.get_Size() << std::endl; */

/*std::vector<std::vector<double>> A = {
    {4, 1, 2, 3},
    {1, 3, 0, 1},
    {2, 0, 5, 2},
    {3, 1, 2, 6}
};*/

//DataMatrix d = CSVLoader::load_CSV("iris.csv");
//d.remove_Spalte("species");
//Matrix x(d);
//SVD::svd_gramm(x);
//Matrix a(A);
//auto[sub,diag,tau, matrix] = Utility::tridiagonalize_symmetric_Matrix(a);
//SVD::svd_gramm(a);
//matrix.print_Matrix();


//std::cout << ActivationFunktion::Linear(10) << std::endl;

//std::cout << ActivationFunktion::Gelu(10) << std::endl;

//std::cout << ActivationFunktion::Sigmoid(10) << std::endl;

/*DataMatrix d = CSVLoader::load_CSV("cutouts.csv", false);
d.remove_Spalte("3072");

Matrix m(d);

StandartScaler s;
Matrix scaled = s.train_apply(m);

PCAFast pca;
Matrix pcaMat = pca.train_apply(scaled);*/


//DataMatrix d = CSVLoader::load_CSV("Auswertung.csv", true, ';');
//DataMatrix spec = d.get_Spalte_as_Matrix_and_Drop("Species");
//spec.print_Matrix();

//std::map<std::string, int> amount;

//auto count = [&amount](std::string type) {++amount[type];};

//spec.get_Spalte<std::string>("Species")->apply_all(count);

//std::println("{}", amount);

//Matrix image_matrix = ImageLoader::loadImagesFromCSV("Auswertung.csv", "D:/Daten/Bat_Orientation_Calls/Bat_Orientation_Calls",3, 1, 0, ';');

//std::cout << image_matrix.get_Size() << std::endl;

//auto[x_train, x_test, y_train, y_test] = TrainTestSplit::split(image_matrix, spec, 0.7, "Species");

//amount.clear();
//y_test.get_Spalte<std::string>("Species")->apply_all(count);

//std::println("{}", amount);
/*
StandartScaler scaler;
Matrix x_train_scaled = scaler.train_apply(x_train);
Matrix x_test_scaled = scaler.apply(x_test);

PCAFast pca;
Matrix x_train_pca = pca.train_apply(x_train_scaled);
Matrix x_test_pca = pca.apply(x_test_scaled);

std::cout << x_train_pca.get_Size() << std::endl;

/*SVM<std::string, LinearKernel> svm(1.0, 1e-3, 10, LinearKernel());
    svm.train(x_train_pca, y_train);
    DataMatrix test_res_vec_svm = svm.predict(x_test_pca);

    std::cout << "Accuracy SVM: " << Metrik::accuracy(y_test, test_res_vec_svm) << std::endl;
    test_res_vec_svm.print_Matrix();

    LogisticRegression<std::string> logReg(5000, 0.01, 42);
    logReg.train(x_train_pca, y_train);
    DataMatrix test_res_vec_log = logReg.predict(x_test_pca);

    std::cout << "Accuracy LogReg: " << Metrik::accuracy(y_test, test_res_vec_log) << std::endl;*/

}