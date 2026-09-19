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
                x_Batches[j] = x_vals[numbers[offset]];
                y_Batches[j] = y_vals[numbers[offset]];
            }
            result.push_back({x_Batches, y_Batches});
        }
        return result;
    }
