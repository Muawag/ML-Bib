#include "DataMatrix.h"
#include <set>
#include "NumberDataMatrix.h"

bool check_Spalten_size(const std::vector<std::unique_ptr<Spalte>>& vec, std::size_t anzahl) {
    for(auto& el : vec) {
        if (el->size() != anzahl)  {
            return false;
        }
    }
    return true;
}

DataMatrix::DataMatrix(const std::vector<std::string>& titles, std::vector<std::unique_ptr<Spalte>>&& spalten) {
    if(titles.size() != spalten.size()) {
        std::cerr << "Titelanzahl und Spaltenanzahl stimmen nicht überein" << std::endl;
        return;
    }
    eintraegeCount_ = spalten[0]->size();
    if(!check_Spalten_size(spalten, eintraegeCount_)) {
        std::cerr << "Elementanzahl der Spalten muss gleich sein" << std::endl;
        return;
    }
    spaltenCount_ = titles.size();
    for(int i = 0; i < spaltenCount_; ++i) {
        if(header_indexes_.contains(titles[i])){
            std::cerr << "Mehmals derselbe Titel" << std::endl;
            return;
        }
        header_indexes_.emplace(titles[i], i);
        spalten_.push_back(std::move(spalten[i]));
    }
    std::copy(titles.begin(), titles.end(), std::back_inserter(header_));
    update_Matrix_Size();
}

void DataMatrix::print_Matrix_Spalten() const {
    for(int i = 0; i < spaltenCount_; ++i) {
        std::cout << header_[i] << std::endl;
        spalten_[i]->print_Spalte();
    }
}
bool DataMatrix::add_Spalte(const std::vector<std::string>& titles, std::vector<std::unique_ptr<Spalte>>&& spalten) {
    if(titles.size() != spalten.size()) {
        std::cerr << "Anzahl Spaltennamen und Spalten passt nich zusammen" << std::endl;
        return false;
    }
    if(!check_Spalten_size(spalten, eintraegeCount_)) {
        std::cerr << "Elementanzahl der Spalten muss gleich sein" << std::endl;
        return false;
    }
        for(int i = 0; i < titles.size(); ++i) {
            if(header_indexes_.contains(titles[i])) {
                std::cerr << "Doppelter spaltenname" << std::endl;
                return false;
            }
            header_.push_back(titles[i]);
            header_indexes_.emplace(titles[i] , (int)header_.size()-1);
            spalten_.push_back(std::move(spalten[i]));
        }
    spaltenCount_ += titles.size();
    update_Matrix_Size();
    return true;
}
void DataMatrix::show_Spalten_info() const {
    for(int i = 0; i < spaltenCount_ ; ++i) {
        std::cout << header_[i] << " Number: " << (spalten_[i]->is_Number() ? "Ja" : "Nein") << std::endl;
    }
}
bool DataMatrix::remove_Spalte(const std::string& name) {
    if(std::find(header_.begin(), header_.end(), name) == header_.end()) {
        std::cerr << "Spalte nicht gefunden" << std::endl;
        return false;
    }
    std::size_t offset = header_indexes_.at(name);
    header_.erase(header_.begin() + offset);
    spalten_.erase(spalten_.begin() + offset);
    header_indexes_.erase(name);
    for(auto it = header_indexes_.begin(); it != header_indexes_.end(); ++it) {
        if(it->second > offset) {
            it->second--;
        }
    }
    spaltenCount_--;
    update_Matrix_Size();
    return true;
}
void DataMatrix::print_Matrix() const {
    for(const std::string& name : header_) {
        std::cout << name << "\t";
    }
    std::cout << std::endl;
    for(int i = 0; i < eintraegeCount_; ++i) {
        for(int j = 0; j < spaltenCount_; ++j) {
            spalten_[j]->print_Element(i);
        }
        std::cout << std::endl;
    }
}
bool DataMatrix::add_Spalte_from_existing(const std::string& title, std::unique_ptr<Spalte>&& spalte_neu) {
    if(std::find(header_.begin(), header_.end(), title) != header_.end()) {
        std::cout << "Doppelter Spaltenname" << std::endl;
        return false;
    }
    header_indexes_.emplace(title,spaltenCount_);
    header_.push_back(title);
    spalten_.push_back(std::move(spalte_neu));
    spaltenCount_++;
    update_Matrix_Size();
    return true;
}
bool DataMatrix::only_Numbers() const {
    for(auto& spalte : spalten_) {
        if(!spalte->is_Number()) return false;
    }
    return true;
}

void DataMatrix::print_header_indexes() const {
    for(auto [text, index] : header_indexes_) {
        std::cout << text << " " << index << std::endl;
    }
}

int DataMatrix::get_Spaltencount() const {
    return spaltenCount_;
}

int DataMatrix::get_Eintraegecount() const {
    return eintraegeCount_;
}

void DataMatrix::remove_if(std::function<bool(int zeilenindex)> predicate) {
    for(int i = eintraegeCount_ -1; i >= 0; --i) {
        if(predicate(i)) {
            for(auto& spalte : spalten_) {
                spalte->remove_Element(i);
            }
            --eintraegeCount_;
        }
    }
    update_Matrix_Size();
}

const std::vector<std::string>& DataMatrix::get_Header() const {
    return header_;
}

void DataMatrix::update_Matrix_Size() {
    matrix_size_.rows = eintraegeCount_;
    matrix_size_.cols = spaltenCount_;
}
const std::vector<std::unique_ptr<Spalte>>& DataMatrix::get_Raw() const {
    return spalten_;
}

Matrix_Size DataMatrix::get_Size() const {
    return matrix_size_;
}