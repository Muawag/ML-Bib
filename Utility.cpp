#include "Utility.h"

    std::vector<std::pair<Matrix, Matrix>> Utility::get_Batches(const Matrix& x_vals, const Matrix& y_vals, int size) {
        if(size > x_vals.get_Size().rows) {
            std::cout << "Batchsize grösser als Trainigsdaten" << std::endl;
            return {{x_vals, y_vals}};
        }
        int batch_count = x_vals.get_Size().rows / size;
        std::vector<std::pair<Matrix, Matrix>> result;
        result.reserve(batch_count);
        std::vector<int> numbers;
        for(std::size_t i = 0; i < x_vals.get_Size().rows; ++i) {
            numbers.push_back(i);
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(numbers.begin(), numbers.end(), gen);
        for(std::size_t i = 0; i < batch_count; ++i) {
            Matrix x_Batches(size, x_vals.get_Size().cols);
            Matrix y_Batches(size, y_vals.get_Size().cols);
            for(std::size_t j = 0; j < size; ++j) {
                int offset = i * size + j;
                auto x_src = x_vals[numbers[offset]];
                std::copy(x_src.begin(), x_src.end(), x_Batches[j].begin());

                auto y_src = y_vals[numbers[offset]];
                std::copy(y_src.begin(), y_src.end(), y_Batches[j].begin());
            }
            result.push_back({x_Batches, y_Batches});
        }
        return result;
    }

    Matrix Utility::prepare_SVM_Matrix(const Matrix& y_Daten, const int index) {
        Matrix y_result(y_Daten.get_Size());
        for(std::size_t i = 0; i < y_Daten.get_Size().rows; ++i) {
            y_result[i][0] = (y_Daten[i][0] == index) ? 1: -1;
        }
        return y_result;
    }
