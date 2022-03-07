#ifndef FIND_PATH_H
#define FIND_PATH_H

#include <iostream>
#include <vector>
#include <tuple>
#include <stdio.h>
#include <math.h>
#include "stlastar.h"
#include "MapSearchNode.h"
#include "MapInfo.h"

#define DEBUG_LISTS 1
#define DEBUG_LIST_LENGTHS_ONLY 1


inline std::tuple<std::vector<int>, int> find_path(
    int *start,
    int *end,
    const MapInfo &Map)
{

    // std::cout << "STL A* Search implementation\n(C)2001 Justin Heyes-Jones\n";

    // Our sample problem defines the world as a 2d array representing a terrain
    // Each element contains an integer from 0 to 5 which indicates the cost 
    // of travel across the terrain. Zero means the least possible difficulty 
    // in travelling (think ice rink if you can skate) whilst 5 represents the 
    // most difficult. 255 indicates that we cannot pass.

    // Create an instance of the search class...

    AStarSearch<MapSearchNode> astarsearch;

    unsigned int SearchCount = 0;

    const unsigned int NumSearches = 1;

    // full path
    std::vector<int> path_full;
    // a short path only contains path corners
    std::vector<int> path_short;
    // how many steps used for this path
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

    #if DEBUG_LISTS

            std::cout << "Steps:" << SearchSteps << "\n";

            int len = 0;

            std::cout << "Open:\n";
            MapSearchNode *p = astarsearch.GetOpenListStart();
            while( p )
            {
                len++;
    #if !DEBUG_LIST_LENGTHS_ONLY			
                ((MapSearchNode *)p)->PrintNodeInfo();
    #endif
                p = astarsearch.GetOpenListNext();
                
            }

            std::cout << "Open list has " << len << " nodes\n";

            len = 0;

            std::cout << "Closed:\n";
            p = astarsearch.GetClosedListStart();
            while( p )
            {
                len++;
    #if !DEBUG_LIST_LENGTHS_ONLY			
                p->PrintNodeInfo();
    #endif			
                p = astarsearch.GetClosedListNext();
            }

            std::cout << "Closed list has " << len << " nodes\n";
    #endif

        }
        while( SearchState == AStarSearch<MapSearchNode>::SEARCH_STATE_SEARCHING );

        if( SearchState == AStarSearch<MapSearchNode>::SEARCH_STATE_SUCCEEDED )
        {
            // std::cout << "Search found goal state\n";
            MapSearchNode *node = astarsearch.GetSolutionStart();

    #if DISPLAY_SOLUTION
            std::cout << "Displaying solution\n";
    #endif

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

    return {path_short, steps};
}


#endif
