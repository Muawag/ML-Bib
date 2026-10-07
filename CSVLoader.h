#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include "DataMatrix.h"


class CSVLoader {
    public:
        static DataMatrix load_CSV(const std::string path, bool header_vorhanden = true);
        static DataMatrix load_CSV(const std::string path, const std::vector<std::string>& header);
};