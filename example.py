from trajectory import Solver

if __name__ == "__main__":
    solver = Solver(d=7, r=2, q=3, dimensions=3)
    waypoints = [[1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 2]]
    timestamps = [0, 0.5, 2, 3, 4, 6]
    solver.solve(waypoints, timestamps)
    solver.show_path()
