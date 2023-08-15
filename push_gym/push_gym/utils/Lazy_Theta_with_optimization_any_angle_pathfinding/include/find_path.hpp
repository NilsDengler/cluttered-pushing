#ifndef FIND_PATH_HPP
#define FIND_PATH_HPP

#include <iostream>
#include <vector>
#include <array>
#include <tuple>
#include "tileadaptor.hpp"
#include "utility.hpp"

static constexpr float WEIGHT_PATH = 1E8;

inline std::tuple<std::vector<int>, float> find_path(
    int *start,
    int *end,
    const std::vector<int> &Map,
    const int &mapSizeX,
    const int &mapSizeY)
{
    //Instantiating our path adaptor
    //passing the map size, and the map
    Vectori mapSize(mapSizeX, mapSizeY);

    TileAdaptor adaptor(mapSize, Map);
    
    //This is a bit of an exageration here for the weight, but it did make my performance test go from 8s to 2s
    Pathfinder pathfinder(adaptor, WEIGHT_PATH);

    //The map was edited so we need to regenerate teh neighbors
    // pathfinder.generateNodes();

    //doing the search
    //merly to show the point of how it work
    //as it would have been way easier to simply transform the vector to id and pass it to search
    // std::tuple<std::vector<int>, float> result_tuple = pathfinder.search(start[1]*mapSizeX+start[0], end[1]*mapSizeX+end[0], adaptor.mMapSize);

    return pathfinder.search(start[1]*mapSizeX+start[0], end[1]*mapSizeX+end[0], adaptor.mMapSize);
}


#endif