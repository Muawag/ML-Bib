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
#include "ImageLoader.h"



        Matrix ImageLoader::loadImagesFromCSV(std::string csvPath,std::string folder, int channels, int idColumn, int imageColumn, char sep, bool has_header){
            int x,y,n;
            std::ifstream csv(csvPath);
            std::vector<std::vector<std::string>> CSVgrid = get_CSV_as_Matrix(csvPath, sep);  // write csv into grid
            std::vector<std::vector<double>> Matrixgrid;
            
            //std::cout<<"b"<<std::endl;

            for (size_t j = has_header ? 1 : 0; j<CSVgrid[idColumn].size();j++) {
                std::vector<double> row;

                // Bilder sind aus irgendeinem grund in png und nicht in jpg
                std::filesystem::path file = CSVgrid[imageColumn][j];
                file.replace_extension(".png");
                std::string path = folder + "/" + file.string();

                //unsigned char *data = stbi_load((folder+"/"+CSVgrid[imageColumn][j]).data(), &x, &y, &n, channels);
                
                unsigned char *data = stbi_load(path.c_str(), &x, &y, &n, channels);

                if(data==nullptr){
                    std::cout<<"failed to find image: "<< folder+"/"+CSVgrid[imageColumn][j] << std::endl;
                    continue;
                }

                // row.push_back(std::stod(CSVgrid[idColumn][j])); // insert id
                


                for(int i=0; i<channels*x*y; i++){
                    row.push_back((double)data[i]); // collect data from image
                }

                
                //std::cout<<"r"<<std::endl;
                Matrixgrid.push_back(row);
                stbi_image_free(data);
            }

            //std::cout<<"s"<<std::endl;

        return Matrix(Matrixgrid);    
        }





