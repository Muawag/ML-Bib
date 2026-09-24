#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "DataMatrix.h"
#include "Matrix_Size.h"

class TrainTestSplit {
    public:
        static Train_Test_Split_Matricies split(DataMatrix& data_matrix, float train_size);
        static Train_Test_Split_Matricies split(DataMatrix& data_matrix, float train_size, std::string stratify_column);
        static Train_Test_Split_Matricies split(DataMatrix& data_matrix, float train_size, std::size_t column_index);

};

struct Train_Test_Split_Matricies {
    DataMatrix train;
    DataMatrix test;
};