#ifndef LINE_OF_SIGHT_H
#define LINE_OF_SIGHT_H


#include <iostream>
#include <algorithm>
#include <cmath>
#include "MapInfo.h"


/*
line_of_sight() checks if the line which passes (x1,y1) and (x2,y2) collides the obstacle
cell (x0,y0) by computing the distance between the line and (x0,y0).
If the distance >= sqrt(2)/2, i.e., there is no intersection between the circumscribed
circle of the square cell and the line, then this cell doesn't block the line-of-sight.

Input:
    previousPoint: {column_x, row_y}, denoted as {x1, y1}
    currentPoint: {column_x, row_y}, denoted as {x2, y2}
    Map: a struct MapInfo

Output:
    block_flag: true if there is at least one obstacle cell blocks the line-of-sight.
                false if the line-of-sight is clear.
*/


inline bool line_of_sight(
    int *previousPoint,
    int *currentPoint,
    const MapInfo &Map)
{
    int min_x = std::min(previousPoint[0], currentPoint[0]);
    int max_x = std::max(previousPoint[0], currentPoint[0]);
    int min_y = std::min(previousPoint[1], currentPoint[1]);
    int max_y = std::max(previousPoint[1], currentPoint[1]);
    // true if there is at least one cell blocks the line-of-sight
    bool block_flag = false;
    // column index is x0, row index is y0
    for (int idx_row = min_y; idx_row < max_y+1; ++idx_row) {
        for (int idx_col = min_x; idx_col < max_x+1; ++idx_col) {
            
            if (block_flag) return block_flag;

            else {
                int cell_value = Map.world_map[(idx_row*Map.map_width)+idx_col];
                if (cell_value == 255) {
                    float distance = std::abs((currentPoint[0]-previousPoint[0]) * (previousPoint[1]-idx_row) - (previousPoint[0]-idx_col) * (currentPoint[1]-previousPoint[1])) / 
                        std::sqrt(std::pow(currentPoint[0]-previousPoint[0], 2) + std::pow(currentPoint[1]-previousPoint[1], 2));
                    if (distance < 0.7071) block_flag = true;
                }
            }
        }
    }
    // the program will enter here if no block at all or only the last cell blocks
    return block_flag;
}


#endif