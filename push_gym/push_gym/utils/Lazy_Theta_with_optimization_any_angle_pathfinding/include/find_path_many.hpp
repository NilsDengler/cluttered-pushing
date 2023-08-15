#ifndef FIND_PATH_MANY_HPP
#define FIND_PATH_MANY_HPP

#include <iostream>
#include <vector>
#include <array>
#include <tuple>
#include "tileadaptor.hpp"
#include "utility.hpp"
#include "get_combination.hpp"

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
static constexpr float WEIGHT_PATH = 1E8;


inline std::tuple<std::vector<std::vector<int>>, std::vector<float>> find_path_many(
    int *agent_position,
    std::vector<int> &targets_position,
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


#endif