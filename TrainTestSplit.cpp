#include "TrainTestSplit.h"
#include <numeric>
#include <algorithm>
#include <random>
#include <memory>
#include <map>



Train_Test_Split_Matricies TrainTestSplit::split(DataMatrix& data_matrix, float train_size, unsigned seed) {
    std::size_t n = data_matrix.get_Size().rows;

    std::vector<std::size_t> indicies(n);
    std::iota(indicies.begin(), indicies.end(), 0);

    std::mt19937 rng(seed);
    std::shuffle(indicies.begin(), indicies.end(), rng);
    std::size_t train_count = static_cast<std::size_t>(n * train_size);

    std::vector<std::size_t> train_indicies(indicies.begin(), indicies.begin() + train_count);
    std::vector<std::size_t> test_indicies(indicies.begin()+ train_count, indicies.end());
    
    std::vector<std::unique_ptr<Spalte>> train_vecs;
    std::vector<std::unique_ptr<Spalte>> test_vecs;

    for(const auto& col : data_matrix.get_Raw()) {
        train_vecs.push_back(col->select_rows(train_indicies));
        test_vecs.push_back(col->select_rows(test_indicies));;
    }
    DataMatrix train_data_matrix(data_matrix.get_Header(), std::move(train_vecs));
    DataMatrix test_data_matrix(data_matrix.get_Header(), std::move(test_vecs));
    return {std::move(train_data_matrix), std::move(test_data_matrix)};
}

Train_Test_Split_Matricies TrainTestSplit::split(DataMatrix& data_matrix, float train_size, std::string stratify_column, unsigned seed) {
    std::size_t n = data_matrix.get_Size().rows;

    std::vector<std::string> string_vec = data_matrix.get_Column_as_String_vec(stratify_column);

    std::map<std::string, std::vector<std::size_t>> indices_per_class;

    for (std::size_t i = 0; i < string_vec.size(); ++i) {
        indices_per_class[string_vec[i]].push_back(i);
    }
}