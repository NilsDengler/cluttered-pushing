#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include "tileadaptor.hpp"
#include "find_path.hpp"


int main()
{
    constexpr int mapSizeX = 70; // width
    constexpr int mapSizeY = 20; // length

    // initialize the map, 0 means no obstacles; each sub vector is a column.
    std::vector<std::vector<int>> Map(mapSizeX, std::vector<int> (mapSizeY, 0));

    // This is a lambda function to create the wall
    auto makeWall = [&Map](const Vectori& pos, const Vectori& size)
    {
        for(int x = 0; x < size.x; x++)
        {
            for(int y = 0; y < size.y; y++)
            {
                Map[pos.x + x][pos.y + y] = 255;
            }
        }
    };

    //borders
    makeWall({0, 0}, {mapSizeX, 1});
    makeWall({0, 0}, {1, mapSizeY});
    makeWall({0, mapSizeY - 1}, {mapSizeX, 1});
    makeWall({mapSizeX - 1, 0}, {1, mapSizeY});

    //walls
    makeWall({5, 0}, {1, mapSizeY - 6});
    makeWall({mapSizeX - 6, 5}, {1, mapSizeY - 6});

    makeWall({mapSizeX - 6, 5}, {4, 1});
    makeWall({mapSizeX - 4, 8}, {4, 1});

    makeWall({20, 0}, {1, mapSizeY - 4});
    makeWall({mapSizeX - 20, 5}, {14, 1});

    int start[2] = {1, 1};
    int end[2] = {mapSizeX - 2, mapSizeY - 2};

    //start and end point
    Map[start[0]][start[1]] = 0;
    Map[end[0]][end[1]] = 0;

    // a conversion from 2D (column major) map to a 1D (row major) map
    std::vector<int> Map_1D;
    Map_1D.reserve(mapSizeX * mapSizeY);
    for(int y = 0; y < mapSizeY; y++)
    {
        for(int x = 0; x < mapSizeX; x++)
        {
            Map_1D.push_back(Map[x][y]);
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // solve it
    auto [path, distance] = find_path(start, end, Map_1D, mapSizeX, mapSizeY);


    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
    std::cout << "Time used [microseconds]:" << duration.count() << std::endl;

    std::cout << "This is the path:" << std::endl;
    for (unsigned long idx=0; idx < path.size(); idx=idx+2)
    {
        std::cout << path[idx] << ", " << path[idx+1] << std::endl;
    }

    std::cout << "Total distance is: " << distance << std::endl;


    // The following is the visualization

    //If we found a path we just want to remove the first and last node
    //Because it will be at our start and end position
    if(path.size())
    {
        path.pop_back();
        path.pop_back();
        path.erase(path.begin());
        path.erase(path.begin());
    }

    //nodes
    int x = 0;
    for(unsigned long idx=0; idx < path.size(); idx=idx+2)
        Map[path[idx]][path[idx+1]] = 1 + x++;

    //draw map
    for(int y = 0; y < static_cast<int>(Map[0].size()); y++)
    {
        for(int x = 0; x < static_cast<int>(Map.size()); x++)
        {
            if ((start[0] == x) && (start[1] == y))
                std::cout << "S";
            else if ((end[0] == x) && (end[1] == y))
                std::cout << "E";
            else
            {
                if ((Map[x][y] != 255) && (Map[x][y] != 0))
                    std::cout << Map[x][y];
                else if (Map[x][y] == 255)
                    std::cout << "#";
                else
                    std::cout << " ";
            }
        }

        std::cout << std::endl;
    }

    std::cout << "#  = walls" << std::endl;
    std::cout << "S  = start" << std::endl;
    std::cout << "E  = end" << std::endl;
    std::cout << "number = path nodes" << std::endl;

    return 0;
}
