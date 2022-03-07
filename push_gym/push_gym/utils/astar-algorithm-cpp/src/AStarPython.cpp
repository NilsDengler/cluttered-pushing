#include <iostream>
#include <vector>
#include <tuple>
#include <stdio.h>
#include <math.h>
#include "stlastar.h"
#include "MapSearchNode.h"
#include "MapInfo.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "get_combination.h"
#include "find_path.h"
#include "smooth_path.h"
#include <chrono>


inline std::tuple<std::vector<std::vector<int>>, std::vector<int>> FindPathAll(
    std::vector<int> agent_position,
    std::vector<int> targets_position,
    std::vector<int> &world_map,
    int &map_width,
    int &map_height)
{
    struct MapInfo Map;
    Map.world_map = world_map;
    Map.map_width = map_width;
    Map.map_height = map_height;

    int num_targets = targets_position.size()/2;
    std::vector<int> start_goal_pair = get_combination(num_targets+1, 2);
    std::vector<std::vector<int>> path_all;
    std::vector<int> steps_all;
    int start[2];
    int goal[2];

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
        auto [path_short_single, steps_used] = find_path(start, goal, Map);
        path_all.push_back(path_short_single);
        steps_all.push_back(steps_used);
    }

    // return path_all;
    return {path_all, steps_all};
}


inline std::tuple<std::vector<int>, int> FindPath(
    std::vector<int> &start,
    std::vector<int> &end,
    std::vector<int> &world_map,
    int &map_width,
    int &map_height)
{

    // std::cout << "STL A* Search implementation\n(C)2001 Justin Heyes-Jones\n";

    // Our sample problem defines the world as a 2d array representing a terrain
    // Each element contains an integer from 0 to 5 which indicates the cost 
    // of travel across the terrain. Zero means the least possible difficulty 
    // in travelling (think ice rink if you can skate) whilst 5 represents the 
    // most difficult. 255 indicates that we cannot pass.

    // Create an instance of the search class...

    struct MapInfo Map;
    Map.world_map = world_map;
    Map.map_width = map_width;
    Map.map_height = map_height;

    AStarSearch<MapSearchNode> astarsearch;

    unsigned int SearchCount = 0;
    const unsigned int NumSearches = 1;

    // full path
    std::vector<int> path_full;
    // a short path only contains path corners
    std::vector<int> path_short;
    // how many steps used
    int steps = 0;

    while(SearchCount < NumSearches)
    {
        // MapSearchNode nodeStart;
        MapSearchNode nodeStart = MapSearchNode(start[0], start[1], Map);
        MapSearchNode nodeEnd(end[0], end[1], Map);

        // Set Start and goal states
        astarsearch.SetStartAndGoalStates( nodeStart, nodeEnd );

        unsigned int SearchState;
        unsigned int SearchSteps = 0;

        do
        {
            SearchState = astarsearch.SearchStep();
            SearchSteps++;
        }
        while( SearchState == AStarSearch<MapSearchNode>::SEARCH_STATE_SEARCHING );

        if( SearchState == AStarSearch<MapSearchNode>::SEARCH_STATE_SUCCEEDED )
        {
            // std::cout << "Search found goal state\n";
            MapSearchNode *node = astarsearch.GetSolutionStart();
            steps = 0;

            // node->PrintNodeInfo();
            path_full.push_back(node->x);
            path_full.push_back(node->y);

            while (true)
            {
                node = astarsearch.GetSolutionNext();

                if ( !node )
                {
                    break;
                }

                // node->PrintNodeInfo();
                path_full.push_back(node->x);
                path_full.push_back(node->y);

                steps ++;

                /*
                Let's say there are 3 steps, x0, x1, x2. To verify whether x1 is a corner for the path.
                If the coordinates of x0 and x1 at least have 1 component same, and the coordinates of 
                x0 and x2 don't have any components same, then x1 is a corner.

                Always append the second path point to path_full.
                When steps >= 2 (starting from the third point), append the point if it's a corner.
                */

                if ((((path_full[2*steps-4]==path_full[2*steps-2]) || (path_full[2*steps-3]==path_full[2*steps-1])) && 
                    ((path_full[2*steps-4]!=node->x) && (path_full[2*steps-3]!=node->y)) && (steps>=2)) || (steps < 2))
                {
                    path_short.push_back(path_full[2*steps-2]);
                    path_short.push_back(path_full[2*steps-1]);
                }
            }

            // the last two elements
            // This works for both steps>2 and steps <=2
            path_short.push_back(path_full[path_full.size()-2]);
            path_short.push_back(path_full[path_full.size()-1]);

            // std::cout << "Solution steps " << steps << endl;

            // Once you're done with the solution you can free the nodes up
            astarsearch.FreeSolutionNodes();
            
        }
        else if( SearchState == AStarSearch<MapSearchNode>::SEARCH_STATE_FAILED ) 
        {
            std::cout << "Search terminated. Did not find goal state\n";
        }

        // Display the number of loops the search went through
        // std::cout << "SearchSteps : " << SearchSteps << "\n";

        SearchCount ++;

        astarsearch.EnsureMemoryFreed();

    }

    return {path_full, steps};
}


// inline std::tuple<std::vector<int>, int> FindPath_test(
inline std::tuple<std::vector<int>, std::vector<int>, int> FindPath_test(
    std::vector<int> &start,
    std::vector<int> &end,
    std::vector<int> &world_map,
    int &map_width,
    int &map_height)
{

    // std::cout << "STL A* Search implementation\n(C)2001 Justin Heyes-Jones\n";

    // Our sample problem defines the world as a 2d array representing a terrain
    // Each element contains an integer from 0 to 5 which indicates the cost 
    // of travel across the terrain. Zero means the least possible difficulty 
    // in travelling (think ice rink if you can skate) whilst 5 represents the 
    // most difficult. 255 indicates that we cannot pass.

    // Create an instance of the search class...

    struct MapInfo Map;
    Map.world_map = world_map;
    Map.map_width = map_width;
    Map.map_height = map_height;

    AStarSearch<MapSearchNode> astarsearch;

    unsigned int SearchCount = 0;
    const unsigned int NumSearches = 1;

    // full path
    std::vector<int> path_full;
    // a short path only contains path corners
    std::vector<int> path_short;
    // how many steps used
    int steps = 0;

    while(SearchCount < NumSearches)
    {
        // MapSearchNode nodeStart;
        MapSearchNode nodeStart = MapSearchNode(start[0], start[1], Map);
        MapSearchNode nodeEnd(end[0], end[1], Map);

        // Set Start and goal states
        astarsearch.SetStartAndGoalStates( nodeStart, nodeEnd );

        unsigned int SearchState;
        unsigned int SearchSteps = 0;

        do
        {
            SearchState = astarsearch.SearchStep();
            SearchSteps++;
        }
        while( SearchState == AStarSearch<MapSearchNode>::SEARCH_STATE_SEARCHING );

        if( SearchState == AStarSearch<MapSearchNode>::SEARCH_STATE_SUCCEEDED )
        {
            // std::cout << "Search found goal state\n";
            MapSearchNode *node = astarsearch.GetSolutionStart();
            steps = 0;

            // node->PrintNodeInfo();
            path_full.push_back(node->x);
            path_full.push_back(node->y);

            while (true)
            {
                node = astarsearch.GetSolutionNext();

                if ( !node )
                {
                    break;
                }

                // node->PrintNodeInfo();
                path_full.push_back(node->x);
                path_full.push_back(node->y);

                steps ++;

                /*
                Let's say there are 3 steps, x0, x1, x2. To verify whether x1 is a corner for the path.
                If the coordinates of x0 and x1 at least have 1 component same, and the coordinates of 
                x0 and x2 don't have any components same, then x1 is a corner.

                Always append the second path point to path_full.
                When steps >= 2 (starting from the third point), append the point if it's a corner.
                */

                if ((((path_full[2*steps-4]==path_full[2*steps-2]) || (path_full[2*steps-3]==path_full[2*steps-1])) && 
                    ((path_full[2*steps-4]!=node->x) && (path_full[2*steps-3]!=node->y)) && (steps>=2)) || (steps < 2))
                {
                    path_short.push_back(path_full[2*steps-2]);
                    path_short.push_back(path_full[2*steps-1]);
                }
            }

            // the last two elements
            // This works for both steps>2 and steps <=2
            path_short.push_back(path_full[path_full.size()-2]);
            path_short.push_back(path_full[path_full.size()-1]);

            // std::cout << "Solution steps " << steps << endl;

            // Once you're done with the solution you can free the nodes up
            astarsearch.FreeSolutionNodes();
            
        }
        else if( SearchState == AStarSearch<MapSearchNode>::SEARCH_STATE_FAILED ) 
        {
            std::cout << "Search terminated. Did not find goal state\n";
        }

        // Display the number of loops the search went through
        // std::cout << "SearchSteps : " << SearchSteps << "\n";

        SearchCount ++;

        astarsearch.EnsureMemoryFreed();

    }


    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<int> path_output = smooth_path(path_short, Map);
    auto t_stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_stop - t_start);
    std::cout << "smooth path time [microseconds]: " << duration.count() << std::endl;

    // return {path_short, steps};
    return {path_short, path_output, steps};
}


inline PYBIND11_MODULE(AStarPython, module) {
    module.doc() = "Python wrapper of AStar c++ implementation";

    module.def("FindPath", &FindPath, "Find a collision-free path");
    module.def("FindPathAll", &FindPathAll, "Find all the collision-free paths");

    module.def("FindPath_test", &FindPath_test, "Find a collision-free path (TEST VERSION)");
}
