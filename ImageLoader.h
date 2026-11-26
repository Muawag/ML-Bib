#pragma once
#include <string>
#include "Matrix.h"

class ImageLoader{
    public:
        static Matrix loadImagesFromCSV(std::string csvPath,std::string folder,
             int channels,int idColumn=0, int imageColumn=1, char sep=',');

};