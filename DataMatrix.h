#pragma once
#include <vector>
#include "Spalte.h"
#include <memory>
#include <map>
#include <iterator>
#include <algorithm>
#include <functional>
#include "Matrix_Size.h"
#include "NumberDataMatrix.h"

class DataMatrix {
    public:
        DataMatrix(const std::vector<std::string>& titles , std::vector<std::unique_ptr<Spalte>>&& spalten);
        DataMatrix() = default;
        void print_Matrix_Spalten() const;
        bool add_Spalte(const std::vector<std::string>& titles, std::vector<std::unique_ptr<Spalte>>&& spalten);
        template<typename T> 
        TypedSpalte<T>* get_Spalte(const std::string& name) const {
            if(std::find(header_.begin(), header_.end(), name) == header_.end()) {
                std::cerr << "Spalte nicht gefunden" << std::endl;
                return nullptr;
            }
            return dynamic_cast<TypedSpalte<T>*>(spalten_[header_indexes_.at(name)].get());
        }
        template<typename T> 
        TypedSpalte<T>* get_Spalte(int index) const {
            if(spaltenCount_<=index) {
                std::cerr << "Spalte nicht gefunden" << std::endl;
                return nullptr;
            }
            return dynamic_cast<TypedSpalte<T>*>(spalten_[index].get());
        }
        bool add_Spalte_from_existing(const std::string& title, std::unique_ptr<Spalte>&& spalte_neu);
        template<typename T>
        bool add_Spalte_from_existing(const std::string& title, TypedSpalte<T>&& add) {
            if(std::find(header_.begin(), header_.end(), title) != header_.end()) {
                std::cout << "Doppelter Spaltenname" << std::endl;
                return false;
            }
            header_indexes_.emplace(title,spaltenCount_);
            header_.push_back(title);
            spalten_.push_back(std::make_unique<TypedSpalte<T>>(std::move(add)));
            spaltenCount_++;
            return true;
        }
        template<typename T>
        bool add_Spalte_from_existing(const std::string& title, TypedSpalte<T>& add) {
            if(std::find(header_.begin(), header_.end(), title) != header_.end()) {
                std::cout << "Doppelter Spaltenname" << std::endl;
                return false;
            }
            header_indexes_.emplace(title,spaltenCount_);
            header_.push_back(title);
            spalten_.push_back(std::make_unique<TypedSpalte<T>>(std::move(add)));
            spaltenCount_++;
            return true;
        }
        DataMatrix remove_Spalte(const std::string& name); 
        void show_Spalten_info() const;
        void print_Matrix() const;
        bool only_Numbers() const;
        template<typename T>
        double get_sum(const std::string& title) const {
            TypedSpalte<T>* temp = get_Spalte<T>(title);
            double sum = 0.0;
            if(temp->is_Number()) {
                for(int i = 0; i < eintraegeCount_; ++i) {
                    sum += (*temp)[i];
                }
                
            }
            return sum;
        }
        template<typename T>
        double get_sum(int index) const {
            TypedSpalte<T>* temp = get_Spalte<T>(index);
            double sum = 0.0;
            if(temp->is_Number()) {
                for(int i = 0; i < eintraegeCount_; ++i) {
                    sum += (*temp)[i];
                }
                
            }
            return sum;
        }
        void print_header_indexes() const ;
        template<typename T>
        double get_mean(const std::string& title) const {
            return get_sum<T>(title) / (int)eintraegeCount_;
        }
        template<typename T>
        double get_mean(int index) const {
            return get_sum<T>(index) / (int)eintraegeCount_;
        }
        int get_Spaltencount() const;
        int get_Eintraegecount() const;
        void remove_if(std::function<bool(int zeilenindex)> predicate);
        const std::vector<std::string>& get_Header() const;
        const std::vector<std::unique_ptr<Spalte>>& get_Raw() const;
        Matrix_Size get_Size() const;
        bool is_Vector() const;
    private:
        void update_Matrix_Size();
    private:
        std::vector<std::string> header_;
        std::vector<std::unique_ptr<Spalte>> spalten_;
        std::map<std::string, int> header_indexes_;
        int spaltenCount_;
        std::size_t eintraegeCount_;
        Matrix_Size matrix_size_;
};