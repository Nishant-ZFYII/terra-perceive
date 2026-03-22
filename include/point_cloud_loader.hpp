#pragma once
#include <Eigen/Dense> //for Eigen::Vector3f
#include <cstdint> //standard int func
#include <vector> 
#include <string> 
#include <fstream> 
#include <iostream> 

std::vector<Eigen::Vector3f> load_bin(const std::string& bin_path);

