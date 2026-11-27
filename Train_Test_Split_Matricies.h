#pragma once
#include "DataMatrix.h"
#include "Matrix.h"

struct Train_Test_Split_Matricies {
    DataMatrix train;
    DataMatrix test;
};

struct Train_Test_Split_Matricies_2 {
    Matrix x_train;
    Matrix x_test;
    DataMatrix y_train;
    DataMatrix y_test;
};