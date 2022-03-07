/*
Adapted from Nikos M.'s answer on https://stackoverflow.com/questions/12991758/creating-all-possible-k-combinations-of-n-items-in-c
Oct. 07, 2020
*/


#include <algorithm>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include "get_combination.h"


int main()
{
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<int> result = get_combination(4, 2);

    auto t1 = std::chrono::high_resolution_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    std::cout << time_used << "microseconds" << std::endl;
    for (unsigned long i = 0; i < result.size(); i = i + 2)
    {
        std::cout << result[i] << ", " << result[i+1] << std::endl;
    }

}