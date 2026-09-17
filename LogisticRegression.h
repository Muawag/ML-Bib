#include <iostream>
#include <vector>
#include "Matrix.h"
#include "Classifier.h"

template <typename T>
class LogisticRegression : public Classifier {
    public:
        std::vector<T> predict(const Matrix& predict_daten) const override {
            std::vector<T> result;
            // Später implementiert
            return result;
        }

        void train(const Matrix& x_daten, const Matrix& y_daten) override {

        }
    
        private:
            
};