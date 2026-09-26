#include "TrainTestSplit.h"



Train_Test_Split_Matricies TrainTestSplit::split(DataMatrix& data_matrix, float train_size) {
    int train_count = data_matrix.get_Size().rows * train_size;
    std::vector<std::unique_ptr<Spalte>> train_vecs;
    std::vector<std::unique_ptr<Spalte>> test_vecs;
}