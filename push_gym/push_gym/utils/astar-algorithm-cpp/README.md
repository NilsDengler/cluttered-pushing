astar-algorithm-cpp
===================

This is a customized version of [astar-algorithm-cpp](https://github.com/justinhj/astar-algorithm-cpp). Given a map and a set of starts and goals, this algorithm can return an optimal path. This repo has very easy-to-build-and-use **C++** implementation and **Python** wrapper.

The following contents starting from **Summary** have been revised such that they fit this forked repo. I really appreciate all the contributors' effort such that I can use this C++ implement of A* algorithm!

Zehui

This repo has been tested with:
* GCC 10.2.0, CMake 3.16.3, Ubuntu 20.04.2 LTS
* GCC 9.3.0, CMake 3.16.3, Ubuntu 20.04.1 LTS
* Clang 12.0.0.0, CMake 3.18.3, macOS 10.15.7

Dependencies
============
For Python:
* [pybind11](https://github.com/pybind/pybind11) If you only install `pybind11` by `pip`, it's possible that CMake can't find it. But you can install it by `apt` or `brew`.
* [numpy](https://numpy.org/).
* [matplotlib](https://matplotlib.org/).


Build
=====
```
$ sudo apt install gcc g++ cmake
$ sudo apt install python3-pybind11 # For macOS: brew install pybind11
$ pip3 install numpy matplotlib
$ git clone https://github.com/zehuilu/astar-algorithm-cpp.git
$ cd <MAIN_DIRECTORY>
$ mkdir build
$ cd build
$ cmake ..
$ make
```


Usage
=====

For C++, the main file is `src/main_sing_path.cpp`.
```
$ cd <MAIN_DIRECTORY>
$ build/main_single
```

For Python, the main file is `test/test_AStarPython.py`.
```
$ cd <MAIN_DIRECTORY>
$ python3 test/test_AStarPython.py
```

Or try `test/solve_plot.py`.
```
$ cd <MAIN_DIRECTORY>
$ python3 test/solve_plot.py
```


Example
=======

**Python**

To call the A Star solver in Python, a simple example is shown below. More details are in `test/solve_plot.py` and `test/test_AStarPython.py`.

```python
import AStarPython
map_width = 20
map_height = 20
# world_map is a 1D list (row-major), 0 means no obstacles, 255 means blocked by obstacles
start = [0, 0] # coordinates for start
goal = [35, 18] # coordinates for goal
# solve it
path_short, steps_used = AStarPython.FindPath(start, end, world_map, Simulator.map_width, Simulator.map_height)
```

Run `test/solve_plot.py`, the result is shown below. Time used is 2.183 ms.
![single path](doc/single_path.png?raw=true "Single Path")


**C++**

To call the A Star solver in C++, a simple example is shown below. More details are in `src/main_single_path.cpp`.

```c++
// ignore all the headers, see more details in src/main_single_path.cpp
int map_width = 20;
int map_height = 20;
int start[2] = {0, 0}; // Create a start state
int end[2] = {18, 14}; // Define the goal state
struct MapInfo Map;
Map.world_map = world_map;
Map.map_width = map_width;
Map.map_height = map_height;
// world_map is a std::vector<int>, 0 means no obstacles, 255 means blocked by obstacles
// solve it
auto [path_short, steps_used] = find_path(start, end, Map);
```

Run `src/main_single_path.cpp`, the path result is shown on the console. Time used is 0.135 ms.
```
This is the short path. Steps used:34
0,0
19,0
19,10
18,10
18,14
```
===================================================================================


Summary
=======

This code is an efficient implementation in C++ and C# of the A* algorithm, designed to be used from high performance realtime applications (video games) and with an optional fast memory allocation scheme.   

It accompanies this A* tutorial: http://www.heyes-jones.com/astar.html

The A* algorithm was described in the paper https://ieeexplore.ieee.org/document/4082128 by Hart, Nillson and Raphael. 
Sadly Nils Nillson passed away in 2019, his work is much appreciated.

Contributions: 

* @justinhj Original code and tutorial
* @ScaryG C# port
* @Rasoul for submitting the path to Bucharest. Sample from Artificial Intelligence: A Modern Approach 
* @sergiosota For fixing a number of issues related to memory management

License
=======

This software is released under the MIT License, see license.txt

Commercial Use
==============

This software has been used in AAA video games and is well tested in the wild. Please let me know if you use this code in you games, studies or hobby projects. 

If you feel the need to pay money for this code, it is not required by the license, but you could contribute to Unicef, a charity which helps children worldwide,  http://www.unicef.org/ that would be awesome.

Projects using this code
========================

If you wish to be added to the list of known products/educational projects using the code please contact me.

* Gun, Activision
* Company of Heroes and Company of Heroes Online, Relic Entertainment
* Angel Engine, a game prototyping engine http://code.google.com/p/angel-engine/
* War of Sonria, a strategy war game on PSP and Playstation 3 by Playground Squad
* Lighthouses AI contest https://github.com/marcan/lighthouses_aicontest
* Self-Driving Car Engineer Nanodegree Program https://github.com/vanAken/CarND-Path-Planning-Project

Introduction
============

This implementation is intended to be simple to read yet fairly
efficient. To build it you can compile, with any recent C++ compiler,
the following files :

For path finder 
* findpath.cpp
* stlastar.h
* MapSearchNode.h
* MapInfo.h
* optionally fsa.h

pathfind has no arguments. You can edit the simple map in pathfind.cpp and the start 
and goal co-ordinates to experiement with the pathfinder.

Fixed size allocator notes: As mentioned briefly in the tutorial you can enable and disable the
faster memory allocation. This allocates a fixed size block of memory, so you have to specify this size
with the astar constructor. You need to enlarge it if you hit an out of memory assert during the
search.

Please let me know if it doesn't work for you and I will try to help. I cannot help if you are using
an old compiler such as Turbo C++, since I update the code to meet Ansi Standard C++ as required.


Cheers!

Justin