#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include "DataMatrix.h"


class CSVLoader {
    public:
        static DataMatrix load_CSV(const std::string path);
        static DataMatrix load_CSV(const std::string path, const std::vector<std::string>& header);
};