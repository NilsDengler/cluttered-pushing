#include <iostream>
#include <vector>
#include <tuple>
#include <array>
#include <chrono>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tileadaptor.hpp"
#include "utility.hpp"
#include "get_combination.hpp"

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
static constexpr float WEIGHT_PATH = 1E8;

inline std::tuple<std::vector<std::vector<int>>, std::vector<float>> FindPathMany(
    std::vector<int> agent_position,
    std::vector<int> targets_position,
    const std::vector<int> &Map,
    const int &mapSizeX,
    const int &mapSizeY)
{
    std::vector<int> start_goal_pair = get_combination(targets_position.size()/2 + 1, 2);
    std::vector<std::vector<int>> path_many;
    std::vector<float> distances_many;
    int start[2];
    int goal[2];

    //Instantiating our path adaptor
    //passing the map size, and the map
    Vectori mapSize(mapSizeX, mapSizeY);
    TileAdaptor adaptor(mapSize, Map);
    //This is a bit of an exageration here for the weight, but it did make my performance test go from 8s to 2s
    // Pathfinder pathfinder(adaptor, 100.f /*weight*/);
    Pathfinder pathfinder(adaptor, WEIGHT_PATH);

    for (unsigned long idx = 0; idx < start_goal_pair.size(); idx = idx + 2)
    {
        int start_idx = start_goal_pair[idx];
        int goal_idx = start_goal_pair[idx+1];

        if (start_idx != 0)
        {
            start[0] = targets_position[2*(start_idx-1)];
            start[1] = targets_position[2*(start_idx-1)+1];
        }
        else
        {
            start[0] = agent_position[0];
            start[1] = agent_position[1];
        }

        if (goal_idx != 0)
        {
            goal[0] = targets_position[2*(goal_idx-1)];
            goal[1] = targets_position[2*(goal_idx-1)+1];

        }
        else
        {
            goal[0] = agent_position[0];
            goal[1] = agent_position[1];
        }

        //doing the search
        auto [Path, Distance] = pathfinder.search(start[1]*mapSizeX+start[0], goal[1]*mapSizeX+goal[0], mapSize);
        path_many.push_back(Path);
        distances_many.push_back(Distance);

        // Regenerate the neighbors for next run
        if (likely(idx < start_goal_pair.size()-1))
        {
            pathfinder.generateNodes();
        }
    }
    return {path_many, distances_many};
}


inline std::tuple<std::vector<int>, float> FindPath(
    std::vector<int> &startPoint,
    std::vector<int> &endPoint,
    std::vector<int> &Map,
    int &mapSizeX,
    int &mapSizeY)
{
    //Instantiating our path adaptor
    //passing the map size, and the map
    Vectori mapSize(mapSizeX, mapSizeY);

    TileAdaptor adaptor(mapSize, Map);
    
    //This is a bit of an exageration here for the weight, but it did make my performance test go from 8s to 2s
    Pathfinder pathfinder(adaptor, 100.f /*weight*/);

    //The map was edited so we need to regenerate teh neighbors
    // pathfinder.generateNodes();

    //doing the search
    //merly to show the point of how it work
    //as it would have been way easier to simply transform the vector to id and pass it to search
    // auto [Path, Distance] = pathfinder.search(startPoint[1]*mapSizeX+startPoint[0], endPoint[1]*mapSizeX+endPoint[0], mapSize);

    return pathfinder.search(startPoint[1]*mapSizeX+startPoint[0], endPoint[1]*mapSizeX+endPoint[0], mapSize);
}


inline PYBIND11_MODULE(LazyThetaStarPython, module) {
    module.doc() = "Python wrapper of Lazy Theta Star c++ implementation";

    module.def("FindPath", &FindPath, "Find a collision-free path");
    module.def("FindPathMany", &FindPathMany, "Find all the collision-free paths");
}