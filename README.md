# Procedural Maze Generator & Explorer

A modular maze generation and visualization project built using Python, PyOpenGL, and GLFW.

The project generates a new random maze using an iterative Depth-First Search (DFS) backtracking algorithm and allows real-time exploration through an OpenGL-rendered environment. The codebase is organized into independent modules for maze generation, rendering, input handling, and gameplay logic to support future extensions such as pathfinding algorithms, texture rendering, and modern OpenGL.

---

## Features

### Maze Generation
- Procedural maze generation using iterative DFS
- Stack-based backtracking algorithm
- Perfect maze generation (every cell reachable)
- Random maze generation on every run
- Optional deterministic generation using random seeds
- Configurable maze dimensions

### Rendering
- GLFW window and OpenGL context
- PyOpenGL rendering pipeline
- Hardware-accelerated rendering
- Grid-based maze rendering
- Separate rendering engine independent of maze logic
- Fullscreen support (F11)

### Gameplay
- WASD movement
- Arrow key movement
- Continuous movement while holding keys
- Collision detection against maze walls
- Goal detection
- Move counter
- Game state management

### Software Design
- Modular architecture
- Object-oriented implementation
- Separation of generation, rendering, and gameplay
- Easily extendable codebase
- Independent rendering primitives

---

## Project Structure

```text
maze_project/
│
├── engine/
│   ├── renderer.py
│   ├── maze_renderer.py
│   ├── input_handler.py
│   └── game_state.py
│
├── maze/
│   ├── cell.py
│   ├── generator.py
│   └── solver.py
│
├── config.py
├── main.py
├── requirements.txt
└── test_generator.py
```

---

## Technologies

- Python 3
- PyOpenGL
- GLFW
- OpenGL
- Object-Oriented Programming


---
## 🚀 Installation

Clone the repository

```bash
git clone git@github.com:ParikchitSen/maze_project.git
cd maze_project
```

Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python main.py
```


## Controls

| Key | Action |
|------|--------|
| W A S D | Move Player |
| Arrow Keys | Move Player |
| Hold Key | Continuous Movement |
| F11 | Toggle Fullscreen |
| ESC | Exit Application |

---

## Algorithms

### Maze Generation

- Iterative Depth-First Search
- Stack-based Backtracking

Time Complexity

```
O(rows × cols)
```

Space Complexity

```
O(rows × cols)
```

---

## Development Roadmap

### Completed

- Iterative DFS maze generation
- OpenGL rendering engine
- Maze renderer
- Player movement
- Collision detection
- Continuous movement
- Game state management
- Random maze generation
- Fullscreen toggle (F11)

### Planned

- Breadth-First Search solver
- A* pathfinding visualization
- Dijkstra visualization
- Maze generation animation
- Multiple maze generation algorithms
- Textured walls
- Improved tile rendering
- Sound effects
- HUD
- Timer
- Score system
- Save/Load mazes
- Modern OpenGL (Shaders, VBOs, VAOs)

---

## Author

**Parikchit Sen**
