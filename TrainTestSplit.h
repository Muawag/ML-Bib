#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "DataMatrix.h"
#include "Matrix_Size.h"
#include "Train_Test_Split_Matricies.h"

class TrainTestSplit {
    public:
        static Train_Test_Split_Matricies split(DataMatrix& data_matrix, float train_size, unsigned seed = 42);
        static Train_Test_Split_Matricies split(DataMatrix& data_matrix, float train_size, std::string stratify_column, unsigned seed = 42);
        //static Train_Test_Split_Matricies split(DataMatrix& data_matrix, float train_size, std::size_t column_index);

};

