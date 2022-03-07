#ifndef SMOOTH_PATH_H
#define SMOOTH_PATH_H


#include <iostream>
#include <vector>
#include <cstdlib>
#include "MapInfo.h"
#include "line_of_sight.h"


/*
Input:
    path: {x0,y0, x1,y1, x2,y2, ...}
    Map: a struct MapInfo

Output:
    path_output:
*/
inline std::tuple<std::vector<int>, bool> smooth_path_once(
    std::vector<int> &path,
    const MapInfo &Map)
{
    std::vector<int> path_output;
    int num_path = path.size(); // each point has two elements
    auto [num_group, remainder] = std::div(num_path/2, 3);

    // true if all line-of-sight are blocked
    // false means that at least one line-of-sight is clear
    bool all_block_flag = true;

    for (int idx_group = 0; idx_group < num_group; ++idx_group) {
        int previousPoint[2] = {path[6*idx_group], path[6*idx_group+1]};
        int currentPoint[2] = {path[6*idx_group+4], path[6*idx_group+5]};
        bool block_flag = line_of_sight(previousPoint, currentPoint, Map);
        all_block_flag &= block_flag;

        // For each 3-point group, if the first one and the third one has a line-of-sight,
        // discard the second one. If not, don't change the group.
        if (!block_flag) {
            path_output.insert(path_output.end(), path.begin()+6*idx_group, path.begin()+6*idx_group+1+1);
            path_output.insert(path_output.end(), path.begin()+6*idx_group+4, path.begin()+6*idx_group+5+1);
        }
        else path_output.insert(path_output.end(), path.begin()+6*idx_group, path.begin()+6*idx_group+5+1);
    }

    int num_path_output = path_output.size();
    // if the path only contains the start and the goal (4 elements), just return it
    if (num_path_output >= 6) {
        if (remainder == 1) {
            int previousPoint[2] = {path_output[num_path_output-4], path_output[num_path_output-3]};
            int currentPoint[2] = {path[num_path-2], path[num_path-1]};
            bool block_flag_last_one = line_of_sight(previousPoint, currentPoint, Map);
            all_block_flag &= block_flag_last_one;

            if (block_flag_last_one) path_output.insert(path_output.end(), path.end()-2, path.end());
            else {
                path_output.erase(path_output.end()-2, path_output.end());
                path_output.insert(path_output.end(), path.end()-2, path.end());
            }
        }
        else if (remainder == 2) {
            int previousPoint[2] = {path_output[num_path_output-2], path_output[num_path_output-1]};
            int currentPoint[2] = {path[num_path-2], path[num_path-1]};
            bool block_flag_last_one = line_of_sight(previousPoint, currentPoint, Map);
            all_block_flag &= block_flag_last_one;

            if (block_flag_last_one) path_output.insert(path_output.end(), path.end()-4, path.end());
            else path_output.insert(path_output.end(), path.end()-2, path.end());
        }
    }

    return {path_output, all_block_flag};
}


inline std::vector<int> smooth_path(
    std::vector<int> &path,
    const MapInfo &Map)
{
    std::vector<int> path_output = path;

    // for debugging
    std::cout << "path.size(): " << path.size() << std::endl;

    while (true) {
        auto _result = smooth_path_once(path_output, Map);
        path_output = std::get<0>(_result);
        auto all_block_flag = std::get<1>(_result);
        if (all_block_flag) break;
    }

    return path_output;
}


#endif