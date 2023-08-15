#pragma once

#include <vector>
#include <array>
#include "utility.hpp"
#include "pathfinding.hpp"
#include <limits>

//The pathfinder is a general algorithm that can be used for mutliple purpose
//So it use adaptor
//This adaptor is for tile grid
class TileAdaptor: public Pathfinder::PathfinderAdaptor
{
public:

    using NodeId = Pathfinder::NodeId;
    using Cost = Pathfinder::Cost;

    const Vectori mMapSize;

    TileAdaptor(const Vectori& mapSize, const std::vector<int> &Map) : mMapSize(mapSize), Map(Map)
    {

    }

    inline virtual size_t getNodeCount() const override
    {
        return mMapSize.x * mMapSize.y;
    }

    //return the distance between two node
    inline virtual Cost distance(const NodeId &n1, const NodeId &n2) const override
    {
        return dist((Vectorf)idToPos(n1), (Vectorf)idToPos(n2));
    }

    //Return true if there is a direct path between n1 and n2
    //Totally not stole this code and did some heavy rewrite
    //The original code was way worse, trust me
    inline virtual bool lineOfSight(const NodeId &n1, const NodeId &n2) const override
    {
        // This line of sight check uses only integer values. First it checks whether the movement along the x or the y axis is longer and moves along the longer
        // one cell by cell. dx and dy specify how many cells to move in each direction. Suppose dx is longer and we are moving along the x axis. For each
        // cell we pass in the x direction, we increase variable f by dy, which is initially 0. When f >= dx, we move along the y axis and set f -= dx. This way,
        // after dx movements along the x axis, we also move dy moves along the y axis.

        Vectori l1 = idToPos(n1);
        Vectori l2 = idToPos(n2);

        Vectori diff = l2 - l1;
//        Vectori l3 = l1;
        Vectori dir; // Direction of movement. Value can be either 1 or -1.

        // The x and y locations correspond to nodes, not cells. We might need to check different surrounding cells depending on the direction we do the
        // line of sight check. The following values are used to determine which cell to check to see if it is unblocked.

        if(diff.y >= 0)
        {
            dir.y = 1;
        }
        else
        {
            diff.y = -diff.y;
            dir.y = -1;
        }

        if(diff.x >= 0)
        {
            dir.x = 1;
        }
        else
        {
            diff.x = -diff.x;
            dir.x = -1;
        }
        
        // x,y as shown in plot. lx,ly are coordinates on the boundayr of cell
        // 0.5 converts the cell index to x-y coordinates
        float lx = l1.x + 0.5 + (float) diff.x / diff.y * dir.x / 2;
        float ly = l1.y + 0.5 + (float) diff.y / diff.x * dir.y / 2;

        if(diff.x >= diff.y)
        { // Move along the x axis and increment/decrement y.
            while(l1.x != l2.x)
            {	//(int) ly will change to next integer when line of sight cross the boundary of cell
                if(!mIsTraversable(l1))
                    return false;
                if((int) ly != l1.y)
                {
                    l1.y += dir.y;
                    if(!mIsTraversable(l1))
	                    return false;
                }
                l1.x += dir.x;
                ly += (float) diff.y / diff.x * dir.y;
            }
        }
        else
        {  //if (diff.x < diff.y). Move along the y axis and increment/decrement x.
            while (l1.y != l2.y)
            {
                if(!mIsTraversable(l1))
                    return false;
                if((int) lx != l1.x)
                {
                    l1.x += dir.x;
                    if(!mIsTraversable(l1))
                    return false;
                }
                l1.y += dir.y;
                lx += (float) diff.x / diff.y * dir.x;   
            }
        }
        return true;
    }

    //return a vector of all the neighbors ids and the cost to travel to them
    //In this adaptor we only need to check the four tileneibors and the cost is always 1
    inline virtual std::vector<std::pair<NodeId, Cost>> getNodeNeighbors(const NodeId &id) const override
    {
        auto pos = idToPos(id);
        const Pathfinder::Cost cost = 1;
        const Pathfinder::Cost not_traversal_cost = 65000;


        std::vector<std::pair<NodeId, Cost>> neighbors;

        //check if we are not on most left if not check if the tile to the left is traversable
        //if so then add it to the neighbor list with its cost(1 for all neighbors)
        if(pos.x != 0)
        {
            if (mIsTraversable({pos.x - 1, pos.y}))
            {
                neighbors.push_back({posToId({pos.x - 1, pos.y}), cost});
            }
            else
            {
                neighbors.push_back({posToId({pos.x - 1, pos.y}), not_traversal_cost});
            }
        }

        if(pos.y != 0)
        {
            if (mIsTraversable({pos.x, pos.y- 1}))
            {
                neighbors.push_back({posToId({pos.x, pos.y - 1}), cost});
            }
            else
            {
                neighbors.push_back({posToId({pos.x, pos.y - 1}), not_traversal_cost});
            }
        }

        if(pos.y != mMapSize.x - 1)
        {
            if (mIsTraversable({pos.x + 1, pos.y}))
            {
                neighbors.push_back({posToId({pos.x + 1, pos.y}), cost});
            }
            else
            {
                neighbors.push_back({posToId({pos.x + 1, pos.y}), not_traversal_cost});
            }
        }

        if(pos.y != mMapSize.y - 1)
        {
            if (mIsTraversable({pos.x, pos.y + 1}))
            {
                neighbors.push_back({posToId({pos.x, pos.y + 1}), cost});
            }
            else
            {
                neighbors.push_back({posToId({pos.x, pos.y  + 1}), not_traversal_cost});
            }
        }

        return neighbors;
    }

    //custom function used to map tile to id
    inline Pathfinder::NodeId posToId(const Vectori &pos) const
    {
        return pos.y * mMapSize.x + pos.x;
    }

    //custom function used to map id to tile
    inline Vectori idToPos(const Pathfinder::NodeId &id) const
    {
        return {static_cast<int>(id % mMapSize.x), static_cast<int>(id / mMapSize.x)};
    }

    //custom function used to map tile to id
    inline Pathfinder::NodeId posToIdArray(const int *pos) const
    {
        return pos[1] * mMapSize.x + pos[0];
    }


private:
    // 1D map
    const std::vector<int> Map;

    inline bool mIsTraversable(const Vectori &vec) const
    {
        return Map[vec.y * mMapSize.x + vec.x] != 255;
    }

};
