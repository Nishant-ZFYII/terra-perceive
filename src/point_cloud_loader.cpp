#include "point_cloud_loader.hpp"

std::vector<Eigen::Vector3f> load_bin(const std::string& bin_path) {
    std::vector<Eigen::Vector3f> points;
    
    //open the file in binary mode
    //std::ios::binary is used to indicate that the file should be read in binary mode
    //this is done to ensure the input data is read as raw bytes and not as a text -> needed for correctly interpreting
    //raw binary data. If this is not set there might be strange behavior when it encounters \end of file or \end of line. 
    //reference : https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/cppBinaryFileIO-2.html

    std::ifstream infile(bin_path, std::ios::binary);

    /* Compue the expected points 
       Each point is represented as 4 float32 values [x,y,z,intensity]*/

    //check if file opens successfully
    if (!infile) {
        std::cerr<<"Error opening file: " <<bin_path <<std::endl;
        return points; //empty Eigen vector
    }
    //move the file pointer to the end of the file 
    //use seekg -> seek get pointer to move the file pointer to the end of the file. 
    infile.seekg(0, std::ios::end);

    //tellg-> returns the current position of the file pointer, curr at the end.
    std::streampos file_size = infile.tellg();

    //move the pointer to the beginning
    infile.seekg(0,std::ios::beg);

    //validate the file size if div by 4 and sizeof(float)
    //compute the number of points in the file

    if (file_size % (4 * sizeof(float)) !=0){
        std::cerr<<"Invalid file size: " <<file_size<<". Expected multiple of " <<4*sizeof(float)<<std::endl;
        return points; //empty Eigen vector
    }
    size_t num_points = file_size / (4 * sizeof(float)); //4 float32 values per point

    //read the data inot the buffer Eigen::Vector3f points
    //reserve space for numpoints to avoid reallocation.
    //the reason for calling reserve is that if the memory runs out, then the operation has to allocate
    //more memory and copy the existing data ; which is O(N) operation. By reserving memory we can avoid this overhead. 
    points.reserve(num_points);

    //loop over the num points and build point vector with x,y,z vals, skip intensity

    for (size_t i = 0; i<num_points; ++i){
        float x,y,z,intensity;
        infile.read(reinterpret_cast<char*>(&x), sizeof(float));
        infile.read(reinterpret_cast<char*>(&y), sizeof(float));
        infile.read(reinterpret_cast<char*>(&z), sizeof(float));
        infile.read(reinterpret_cast<char*>(&intensity), sizeof(float)); //read but ignore

        points.emplace_back(x,y,z);
    }
    infile.close();
    return points;

}