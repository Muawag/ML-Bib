#include "Spalte.h"
#include "DataMatrix.h"
#include "NumberDataMatrix.h"
#include "Scaler.h"
#include "Matrix.h"
#include "SVD.h"
#include "PCA.h"

int main() {
    TypedSpalte<int> spalte({1,2,3,4});
    TypedSpalte<std::string> spalte2({"a","b","c","d"});
    //std::cout << spalte[10] << std::endl;
    std::vector<std::string> titles = {"Spalte1", "Spalte2"};
    std::vector<std::unique_ptr<Spalte>> spalten;
    spalten.emplace_back(std::make_unique<TypedSpalte<int>>(spalte));
    spalten.emplace_back(std::make_unique<TypedSpalte<std::string>>(spalte2));
    DataMatrix d(titles, std::move(spalten));
    //d.print_Matrix();
    std::vector<std::string> tit2 = {"a1","a2"};
    TypedSpalte<double> spalte3({3.2,5.3,4.8,3.4});
    std::vector<std::unique_ptr<Spalte>> spaltenNeu;
    spaltenNeu.emplace_back(std::make_unique<TypedSpalte<int>>(spalte));
    spaltenNeu.emplace_back(std::make_unique<TypedSpalte<double>>(spalte3));
    if(d.add_Spalte(tit2, std::move(spaltenNeu))){
        //std::cout << "passt" << std::endl;
    }
    //d.print_Matrix_Spalten();
    //d.print_Matrix();
    //TypedSpalte<int>* getSpalte = d.get_Spalte<int>("Spalte1");
    //getSpalte->print_Spalte();
    //getSpalte->change_Element(5,1);
    //getSpalte->print_Spalte();
    //d.print_Matrix();
    //d.show_Spalten_info();
    //d.remove_Spalte("a1");
    //d.show_Spalten_info();
    d.remove_Spalte("a1");
    //d.print_Matrix();
    auto typedNeu = *d.get_Spalte<int>(0) + *d.get_Spalte<double>(2);
    //d.add_Spalte_from_existing("test", std::make_unique<decltype(typedNeu)>(typedNeu));
    d.add_Spalte_from_existing("test", std::move(typedNeu));
    auto rein = *d.get_Spalte<int>(0) * 1.5f;
    d.add_Spalte_from_existing("test2", rein);
    //typedNeu.print_Spalte();
    //d.print_Matrix();
    d.remove_if([&](int i) {
        return(*d.get_Spalte<double>("test"))[i] < 5;
    });
    //d.print_Matrix();
    //std::cout << d.get_sum<double>(3) << std::endl;
    //std::cout << d.get_mean<double>("test") << std::endl;
    d.remove_Spalte("Spalte2");
    NumberDataMatrix num_Matrix(d);
    NumberDataMatrix num2 = num_Matrix.transpose();
    NumberDataMatrix num3 = num_Matrix * num2;
    //num_Matrix.print_Matrix();
    //num3.print_Matrix();
    //std::cout << num3.get_Size() << std::endl;
    std::vector<double> testVector = num3.get_Row(1);
    //std::cout << num_Matrix.get_Size() << std::endl;
    std::vector<double> testVector2 = num_Matrix.get_Row(1);
    /*for(double d : testVector) {
        std::cout << d << "\t"; 
    }*/
    StandartScaler scaler;
    scaler.train(num_Matrix);
    NumberDataMatrix num_scaled = scaler.apply(num_Matrix);
    Matrix mat(d);
    Matrix mat2(d);
    //mat.print_Matrix();
    Matrix mat3 = mat * mat2.transpose();
    //mat3.print_Matrix();
    std::vector<std::vector<double>> matrix = {
    {2,  3,  1,  5,  7,  4},
    {4,  1,  7,  2,  3,  8},
    {6,  8,  3,  4,  1,  2},
    {1,  5,  9,  3,  6,  7},
    {7,  2,  4,  6,  8,  1},
    {3,  6,  2,  8,  4,  5},
    {5, 4, 6, 1, 2, 9},
    {8, 7, 5, 3, 9, 6}
    };
    Matrix mat4(matrix);
    //Q_R_Matrizen qr = svd.q_r_step_zeile(mat3);
    //qr.Q.print_Matrix();
    //qr.R.print_Matrix();
    //auto [P, J, Q] = svd.bidiagonalize(mat4);
    //J.print_Matrix();
    /*std::vector<std::vector<double>> matrix_t = {
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 10}
    };
    Matrix einbett(matrix_t);
    Matrix e = einbett.einbetten_u_r(5);
    e.print_Matrix();*/
    //P.print_Matrix();
    //J.print_Matrix();
    //Q.print_Matrix();
    //num_scaled.print_Matrix();
    //Matrix result = P * J * Q.transpose();
    //result.print_Matrix();
    /*auto [X, D, Y] = svd.givens_Iteration(J);
    X.print_Matrix();
    D.print_Matrix();
    Y.print_Matrix();*/
    /*SVD_Matrizen svd_Mats = SVD::svd(mat4);
    svd_Mats.Sigma.print_Matrix();
    svd_Mats.U.print_Matrix();
    svd_Mats.V.print_Matrix();
    Matrix result = svd_Mats.U * svd_Mats.Sigma * svd_Mats.V.transpose();
    result.print_Matrix();*/
    std::vector<std::vector<double>> pca_vector = {
    {2.5, 2.4},
    {0.5, 0.7},
    {2.2, 2.9},
    {1.9, 2.2},
    {3.1, 3.0},
    {2.3, 2.7}
    };
    Matrix pca_matrix(pca_vector);
    PCA pca;
    Matrix result = pca.train_apply(mat4);
    result.print_Matrix();
}