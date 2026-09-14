#pragma once
#include <iostream>

struct Matrix_Size {
    int rows;
    int cols;
};

std::ostream& operator<<(std::ostream& os, const Matrix_Size& m);

bool operator==(const Matrix_Size& size1, const Matrix_Size& m2);