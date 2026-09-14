#pragma once
#include <string>
#include <cstddef>
#include <typeinfo>
#include <vector>
#include <iostream>
#include <concepts>
#include <stdexcept>

class Spalte {
    public:
        virtual ~Spalte() = default;
        virtual std::string typName() const = 0;
        virtual size_t size() const = 0;
        virtual void print_Spalte() const = 0;
        virtual bool is_Number() const = 0;
        virtual void print_Element(int pos) const = 0;
        virtual void remove_Element(int pos) = 0;
        virtual std::vector<double> get_Daten_as_Doubles() const = 0;
        virtual double get_Element_as_Double(std::size_t index) const = 0;
};

template<typename T>
class TypedSpalte : public Spalte {
    public:
        TypedSpalte() = default;
        explicit TypedSpalte(std::vector<T>&& daten) : daten_(std::move(daten)) {}
        explicit TypedSpalte(std::size_t size) : daten_(size) {}
        std::string typName() const override { return typeid(T).name(); }
        size_t size() const override { return daten_.size(); }
        void add_data(const T& newValue) {
            daten_.push_back(newValue);
        }
        T& operator[](size_t index) {
            return daten_.at(index);
        }
        const T& operator[](size_t index) const{
            return daten_.at(index);
        }
        void print_Spalte() const override {
            for (const T& item : daten_)
            {
                std::cout << item << std::endl;
            }
            
        }
        void change_Element(const T& el, int pos) {
            daten_[pos] = el;
        }
        std::vector<T>* get_Raw() {
            return daten_;
        }
        bool is_Number() const override {
            if constexpr (std::is_arithmetic_v<T>) {
                return true;
            }
            return false;
        }
        void print_Element(int pos) const override {
            std::cout << daten_[pos] << "\t";
        }
        template<typename U>
        auto operator+(const TypedSpalte<U>& other) const -> TypedSpalte<std::common_type_t<T,U>> {
            if(other.size() != this->size()) {
                std::cerr << "Vectoren müssen gleich groß sein" << std::endl;
                return {}; 
            }
            using R = std::common_type_t<T,U>;
            TypedSpalte<R> result(this->size());
            for(std::size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] + other[i];
            }
            return result;
        }
        template<typename U> 
        auto operator*(const TypedSpalte<U>& other) const -> TypedSpalte<std::common_type_t<T,U>> {
            if(other.size() != this->size()) {
                std::cerr << "Vectoren müssen gleich groß sein" << std::endl;
                return {}; 
            }
            using R = std::common_type_t<T,U>;
            TypedSpalte<R> result(this->size());
            for(std::size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] * other[i];
            }
            return result;
        }
        template<typename U> 
        auto operator-(const TypedSpalte<U>& other) const -> TypedSpalte<std::common_type_t<T,U>> {
            if(other.size() != this->size()) {
                std::cerr << "Vectoren müssen gleich groß sein" << std::endl;
                return {}; 
            }
            using R = std::common_type_t<T,U>;
            TypedSpalte<R> result(this->size());
            for(std::size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] - other[i];
            }
            return result;
        }
        template<typename U> 
        auto operator/(const TypedSpalte<U>& other) const -> TypedSpalte<std::common_type_t<T,U>> {
            if(other.size() != this->size()) {
                std::cerr << "Vectoren müssen gleich groß sein" << std::endl;
                return {}; 
            }
            using R = std::common_type_t<T,U>;
            TypedSpalte<R> result(this->size());
            for(std::size_t i = 0; i < this->size(); ++i) {
                if(other[i] == 0) {
                    std::cout << "Durch 0 teilen" << std::endl;
                    return {};
                }
                result[i] = (*this)[i] / other[i];
            }
            return result;
        }
        template<typename U>
        requires std::integral<U> || std::floating_point<U>
        auto operator*(U value) const -> TypedSpalte<std::common_type_t<T,U>> {
            using R = std::common_type_t<T,U>;
            TypedSpalte<R> result(this->size());
            for(std::size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] * value;
            }
            return result;
        }
        template<typename U>
        requires std::integral<U> || std::floating_point<U>
        auto operator/(U value) const -> TypedSpalte<std::common_type_t<T,U>> {
            if(value == 0) {
                std::cout << "Durch 0 teilen" << std::endl;
                return {}; 
            }
            using R = std::common_type_t<T,U>;
            TypedSpalte<R> result(this->size());
            for(std::size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] / value;
            }
            return result;
        }
        template<typename U>
        requires std::integral<U> || std::floating_point<U>
        auto operator+(U value) const -> TypedSpalte<std::common_type_t<T,U>> {
            using R = std::common_type_t<T,U>;
            TypedSpalte<R> result(this->size());
            for(std::size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] + value;
            }
            return result;
        }
        template<typename U>
        requires std::integral<U> || std::floating_point<U>
        auto operator-(U value) const -> TypedSpalte<std::common_type_t<T,U>> {
            using R = std::common_type_t<T,U>;
            TypedSpalte<R> result(this->size());
            for(std::size_t i = 0; i < this->size(); ++i) {
                result[i] = (*this)[i] - value;
            }
            return result;
        }
        void remove_Element(int pos) override {
            daten_.erase(daten_.begin() + pos);
        }
        std::vector<double> get_Daten_as_Doubles() const override {
            std::vector<double> num_spalte(daten_.size());
            if constexpr (std::is_arithmetic_v<T>) {
            for(std::size_t i = 0; i < daten_.size(); ++i) {
                num_spalte[i] = static_cast<double>(daten_[i]);
            }
        }
            return num_spalte;
        }
        double get_Element_as_Double(std::size_t index) const override {
            if(index >= daten_.size()) {
                throw std::out_of_range("Index war groesser als die Spalte");
            }
            if constexpr (std::is_arithmetic_v<T>) {
                double ret;
                ret = static_cast<double>(daten_[index]);
                return ret;
            }
            else {
                throw std::runtime_error("Spalte konnte nicht zu double gecastet werden");
            }
        }
    private:
        std::vector<T> daten_;
};