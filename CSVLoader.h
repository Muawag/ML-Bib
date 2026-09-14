#pragma once
#include <iostream>
#include <fstream>
class CSVLoader {
    private:
        int x;
        std::ifstream iFile;
    public:
        //Constructors
        CSVLoader();
        CSVLoader(const std::string& path);
        //CSVLoader(const CSVLoader& csvloader);
        std::ifstream get_File();
};