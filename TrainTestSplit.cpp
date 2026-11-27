#include "TrainTestSplit.h"
#include <numeric>
#include <algorithm>
#include <random>
#include <memory>
#include <map>


Train_Test_Split_Matricies TrainTestSplit::split(DataMatrix& data_matrix, double train_size, unsigned seed) {
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

Train_Test_Split_Matricies TrainTestSplit::split(DataMatrix& data_matrix, double train_size, std::string stratify_column, unsigned seed) {

    std::vector<std::string> string_vec = data_matrix.get_Column_as_String_vec(stratify_column);

    std::map<std::string, std::vector<std::size_t>> indices_per_class;

    for (std::size_t i = 0; i < string_vec.size(); ++i) {
        indices_per_class[string_vec[i]].push_back(i);
    }

    std::mt19937 rng(seed);
    std::vector<std::size_t> train_indices;
    std::vector<std::size_t> test_indices;

    for(auto& [label, vec] : indices_per_class) {
        std::shuffle(vec.begin(), vec.end(), rng);
        std::size_t train_count;
        if(vec.size() == 1) {
            train_count = 1;
        }
        else {
            train_count = static_cast<std::size_t>(vec.size() * train_size);
        }
        std::move(vec.begin(), vec.begin() + train_count, std::back_inserter(train_indices));
        std::move(vec.begin()+ train_count, vec.end(), std::back_inserter(test_indices));
    }

    std::vector<std::unique_ptr<Spalte>> train_vecs;
    std::vector<std::unique_ptr<Spalte>> test_vecs;
    std::shuffle(train_indices.begin(), train_indices.end(), rng);
    std::shuffle(test_indices.begin(), test_indices.end(), rng);

    for(auto& col : data_matrix.get_Raw()) {
        train_vecs.push_back(col->select_rows(train_indices));
        test_vecs.push_back(col->select_rows(test_indices));
    }

    DataMatrix train_data_matrix(data_matrix.get_Header(), std::move(train_vecs));
    DataMatrix test_data_matrix(data_matrix.get_Header(), std::move(test_vecs));
    return {std::move(train_data_matrix), std::move(test_data_matrix)};
}

Train_Test_Split_Matricies TrainTestSplit::split(DataMatrix& data_matrix, double train_size, std::size_t column_index, unsigned seed) {
    if(column_index > data_matrix.get_Header().size()) {
        throw std::runtime_error("Zu grosser Zeilenindex");
    }
    std::string title = data_matrix.get_Header()[column_index];
    return TrainTestSplit::split(data_matrix, train_size, title, seed);
}

Train_Test_Split_Matricies_2 TrainTestSplit::split(Matrix& matrix, DataMatrix& y_labels, double train_size, std::string stratify_column, unsigned seed) {
          
    
    if (train_size < 0.0 || train_size > 1.0) {
        throw std::invalid_argument("train_size muss zwischen 0 und 1 liegen");
    }

    std::vector<std::string> string_vec = y_labels.get_Column_as_String_vec(stratify_column);

    if (string_vec.size() != static_cast<std::size_t>(matrix.get_Size().rows)) {
        throw std::invalid_argument("Matrix und Labels haben unterschiedlich viele Zeilen");
    }

    std::map<std::string, std::vector<std::size_t>> indices_per_class;


    for (std::size_t i = 0; i < string_vec.size(); ++i) {
        indices_per_class[string_vec[i]].push_back(i);
    }

    std::mt19937 rng(seed);
    std::vector<std::size_t> train_indices;
    std::vector<std::size_t> test_indices;


    for(auto& [label, vec] : indices_per_class) {
        std::shuffle(vec.begin(), vec.end(), rng);
        std::size_t train_count;
        if(vec.size() == 1) {
            train_count = 1;
        }
        else {
            train_count = static_cast<std::size_t>(vec.size() * train_size);
        }
        std::move(vec.begin(), vec.begin() + train_count, std::back_inserter(train_indices));
        std::move(vec.begin()+ train_count, vec.end(), std::back_inserter(test_indices));
    }

    std::vector<std::unique_ptr<Spalte>> train_vecs;
    std::vector<std::unique_ptr<Spalte>> test_vecs;
    std::shuffle(train_indices.begin(), train_indices.end(), rng);
    std::shuffle(test_indices.begin(), test_indices.end(), rng);

    for(auto& col : y_labels.get_Raw()) {
        train_vecs.push_back(col->select_rows(train_indices));
        test_vecs.push_back(col->select_rows(test_indices));
    }

    DataMatrix train_data_matrix(y_labels.get_Header(), std::move(train_vecs));
    DataMatrix test_data_matrix(y_labels.get_Header(), std::move(test_vecs));


    // Matrix Split

    const std::size_t cols = matrix.get_Size().cols;

    Matrix x_train(train_indices.size(), cols, 0.0);
    Matrix x_test(test_indices.size(), cols, 0.0);

    auto copy_rows = [&](const std::vector<std::size_t>& indices, Matrix& target) {
        const double* src = matrix.data();
        double* dst = target.data();

        for (std::size_t i = 0; i < indices.size(); ++i) {
            std::copy_n(src + indices[i] * cols, cols, dst + i * cols);
        }
    };

    copy_rows(train_indices, x_train);
    copy_rows(test_indices, x_test);

    return {
        std::move(x_train),
        std::move(x_test),
        std::move(train_data_matrix),
        std::move(test_data_matrix)
    };
}