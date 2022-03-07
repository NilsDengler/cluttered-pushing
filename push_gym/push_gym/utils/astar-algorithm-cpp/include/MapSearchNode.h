////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// STL A* Search implementation
// (C)2001 Justin Heyes-Jones
//
// Finding a path on a simple grid maze
// This shows how to do shortest path finding using A*

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef MAPSEARCHNODE_H
#define MAPSEARCHNODE_H

#include <iostream>
#include <stdio.h>
#include <math.h>
#include "stlastar.h"
#include "MapInfo.h"


class MapSearchNode
{
public:
    int x;	 // the (x,y) positions of the node
    int y;	
    struct MapInfo map;

    MapSearchNode() { x = 0; y = 0; }
    MapSearchNode(int px, int py, const MapInfo &map_input) {
        x = px;
        y = py;
        map = map_input;
    }

    float GoalDistanceEstimate( MapSearchNode &nodeGoal );
    bool IsGoal( MapSearchNode &nodeGoal );
    bool GetSuccessors( AStarSearch<MapSearchNode> *astarsearch, MapSearchNode *parent_node );
    float GetCost( MapSearchNode &successor );
    bool IsSameState( MapSearchNode &rhs );

    void PrintNodeInfo(); 

    inline int GetMap(int x, int y);

};

bool MapSearchNode::IsSameState( MapSearchNode &rhs )
{

    // same state in a maze search is simply when (x,y) are the same
    if( (x == rhs.x) &&
        (y == rhs.y) )
    {
        return true;
    }
    else
    {
        return false;
    }

}

void MapSearchNode::PrintNodeInfo()
{
    char str[100];
    sprintf( str, "Node position : (%d,%d)\n", x,y );

    std::cout << str;
}

// Here's the heuristic function that estimates the distance from a Node
// to the Goal. 

float MapSearchNode::GoalDistanceEstimate( MapSearchNode &nodeGoal )
{
    return abs(x - nodeGoal.x) + abs(y - nodeGoal.y);
}

bool MapSearchNode::IsGoal( MapSearchNode &nodeGoal )
{

    if( (x == nodeGoal.x) &&
        (y == nodeGoal.y) )
    {
        return true;
    }

    return false;
}

// This generates the successors to the given Node. It uses a helper function called
// AddSuccessor to give the successors to the AStar class. The A* specific initialisation
// is done for each node internally, so here you just set the state information that
// is specific to the application
bool MapSearchNode::GetSuccessors( AStarSearch<MapSearchNode> *astarsearch, MapSearchNode *parent_node )
{

    int parent_x = -1; 
    int parent_y = -1; 

    if( parent_node )
    {
        parent_x = parent_node->x;
        parent_y = parent_node->y;
    }


    // MapSearchNode NewNode;

    // push each possible move except allowing the search to go backwards

    if( (GetMap( x-1, y ) < 255) 
        && !((parent_x == x-1) && (parent_y == y))
        ) 
    {
        MapSearchNode NewNode = MapSearchNode( x-1, y, map );
        astarsearch->AddSuccessor( NewNode );
    }	

    if( (GetMap( x, y-1 ) < 255) 
        && !((parent_x == x) && (parent_y == y-1))
        ) 
    {
        MapSearchNode NewNode = MapSearchNode( x, y-1, map );
        astarsearch->AddSuccessor( NewNode );
    }	

    if( (GetMap( x+1, y ) < 255)
        && !((parent_x == x+1) && (parent_y == y))
        ) 
    {
        MapSearchNode NewNode = MapSearchNode( x+1, y, map );
        astarsearch->AddSuccessor( NewNode );
    }	

        
    if( (GetMap( x, y+1 ) < 255) 
        && !((parent_x == x) && (parent_y == y+1))
        )
    {
        MapSearchNode NewNode = MapSearchNode( x, y+1, map );
        astarsearch->AddSuccessor( NewNode );
    }	

    return true;
}

// given this node, what does it cost to move to successor. In the case
// of our map the answer is the map terrain value at this node since that is 
// conceptually where we're moving

float MapSearchNode::GetCost( MapSearchNode &successor )
{
    return (float) GetMap( x, y );

}


inline int MapSearchNode::GetMap(int x, int y)
{
    if( x < 0 ||
        x >= map.map_width ||
            y < 0 ||
            y >= map.map_height
        )
    {
        return 255;	 
    }

    return map.world_map[(y*map.map_width)+x];
}


#endif