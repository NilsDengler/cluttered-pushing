#ifndef MAPINFO_H
#define MAPINFO_H


#include <vector>


struct MapInfo 
{ 
    int map_width;
    int map_height;
    std::vector<int> world_map;
};


#endif