#include "Matrix_Size.h"

std::ostream& operator<<(std::ostream& os, const Matrix_Size& m) {
    os << "Rows: " << m.rows << "\n" << "Cols: " << m.cols << std::endl;
    return os;
}

bool operator==(const Matrix_Size& m1, const Matrix_Size& m2) {
    return (m1.cols == m2.cols && m1.rows == m2.rows);
}