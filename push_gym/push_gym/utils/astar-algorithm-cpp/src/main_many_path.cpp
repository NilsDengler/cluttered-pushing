#include <iostream>
#include <vector>
#include <chrono>
#include "MapInfo.h"
#include "find_path.h"
#include "find_path_all.h"


// Main
int main()
{
    std::vector<int> world_map = 
    {
    // 0001020304050607080910111213141516171819
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,   // 00
        1,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,1,   // 01
        1,9,9,1,1,9,9,9,1,9,1,9,1,9,1,9,9,9,1,1,   // 02
        1,9,9,1,1,9,9,9,1,9,1,9,1,9,1,9,9,9,1,1,   // 03
        1,9,1,1,1,1,9,9,1,9,1,9,1,1,1,1,9,9,1,1,   // 04
        1,9,1,1,9,1,1,1,1,9,1,1,1,1,9,1,1,1,1,1,   // 05
        1,9,9,9,9,1,1,1,1,1,1,9,9,9,9,1,1,1,1,1,   // 06
        1,9,9,9,9,9,9,9,9,1,1,1,9,9,9,9,9,9,9,1,   // 07
        1,9,1,1,1,1,1,1,1,1,1,9,1,1,1,1,1,1,1,1,   // 08
        1,9,1,9,9,9,9,9,9,9,1,1,9,9,9,9,9,9,9,1,   // 09
        1,9,1,1,1,1,9,1,1,9,1,1,1,1,1,1,1,1,1,1,   // 10
        1,9,9,9,9,9,1,9,1,9,1,9,9,9,9,9,1,1,1,1,   // 11
        1,9,1,9,1,9,9,9,1,9,1,9,1,9,1,9,9,9,1,1,   // 12
        1,9,1,9,1,9,9,9,1,9,1,9,1,9,1,9,9,9,1,1,   // 13
        1,9,1,1,1,1,9,9,1,9,1,9,1,1,1,1,9,9,1,1,   // 14
        1,9,1,1,9,1,1,1,1,9,1,1,1,1,9,1,1,1,1,1,   // 15
        1,9,9,9,9,1,1,1,1,1,1,9,9,9,9,1,1,1,1,1,   // 16
        1,1,9,9,9,9,9,9,9,1,1,1,9,9,9,1,9,9,9,9,   // 17
        1,9,1,1,1,1,1,1,1,1,1,9,1,1,1,1,1,1,1,1,   // 18
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,   // 19
    };

    for (size_t idx = 0; idx < world_map.size(); ++idx) {
        if (world_map[idx] == 9)
            world_map[idx] = 255;
        else
            world_map[idx] = 0;
    }

    int map_width = 20;
    int map_height = 20;
    struct MapInfo Map;
    Map.world_map = world_map;
    Map.map_width = map_width;
    Map.map_height = map_height;

    int agent_position[2] = {0, 0}; // Define the position for the agent
    std::vector<int> targets_position{0,19, 19,19, 19,0}; // Define a set of targets positions [x0,y0, x1,y1, x2,y2]

    auto start_time = std::chrono::high_resolution_clock::now();


    auto [path_all, steps_all] = find_path_all(agent_position, targets_position, Map);


    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
    std::cout << "Time used [microseconds]:" << duration.count() << std::endl;


    for(long unsigned int i=0; i<path_all.size(); i++)
    {
        std::cout << "This is a path, idx = " << i  << ". Steps used:" << steps_all[i] << std::endl;
        if (path_all[i].size() > 0)
        {
            for (long unsigned int j=0; j<path_all[i].size(); j = j + 2)
            {
                std::cout << path_all[i][j] << "," << path_all[i][j+1] << std::endl;
            }
        }
        else
        {
            std::cout << "This is an empty path. No path!" << std::endl;
        }
        
    }

    return 0;
}