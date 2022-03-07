/*
Adapted from Nikos M.'s answer on https://stackoverflow.com/questions/12991758/creating-all-possible-k-combinations-of-n-items-in-c
Oct. 07, 2020
*/
#ifndef GET_COMBINATION_H
#define GET_COMBINATION_H


#include <algorithm>
#include <iostream>
#include <string>
#include <vector>


inline std::vector<int> get_combination(int N, int K)
{
    std::vector<int> result;
    int idx = 0;

    std::string bitmask(K, 1); // K leading 1's
    bitmask.resize(N, 0); // N-K trailing 0's

    // print integers and permute bitmask
    do {
        for (int i = 0; i < N; ++i) // [0..N-1] integers
        {
            if (bitmask[i]) result.push_back(i);
        }
        idx++;
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

    return result;
}


#endif