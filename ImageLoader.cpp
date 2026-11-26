#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include "libs/stb_image.h" // https://github.com/nothings/stb/blob/master/stb_image.h
#include <iostream>
#include "Matrix.h"
#include <filesystem>
#include <string>
#include <vector>
#include <fstream>
#include "CSVLoader.h"




class ImageLoader{

    public:
        


        Matrix loadImagesFromCSV(std::string csvPath,std::string folder, int channels,int idColumn, int imageColumn){
            int x,y,n;
            std::ifstream csv(csvPath);
            std::vector<std::vector<std::string>> CSVgrid =get_CSV_as_Matrix(csvPath);  // write csv into grid
            std::vector<std::vector<double>> Matrixgrid;
            
            std::cout<<"b"<<std::endl;

            for (int j=0; j<CSVgrid[idColumn].size();j++) {
                std::vector<double> row;
                unsigned char *data = stbi_load((folder+"/"+CSVgrid[imageColumn][j]).data(), &x, &y, &n, channels);
                
                if(data==nullptr){
                    std::cout<<"failed to find image: "<< folder+"/"+CSVgrid[imageColumn][j] << std::endl;
                    continue;
                }

                row.push_back(std::stod(CSVgrid[idColumn][j])); // insert id
                


                for(int i=0; i<channels*x*y; i++){
                    row.push_back((double)data[i]); // collect data from image
                }

                
                std::cout<<"r"<<std::endl;
                Matrixgrid.push_back(row);
                stbi_image_free(data);
            }

            std::cout<<"s"<<std::endl;

        return Matrix(Matrixgrid);    
        }

};



