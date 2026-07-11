from maze.generator import MazeGenerator

maze = MazeGenerator(10, 10)
maze.generate()

maze.print_ascii()   # or whatever method you wrote
