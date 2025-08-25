from trajectory import TimestampSolver

if __name__ == "__main__":
    solver = TimestampSolver(d=5, r=2, q=2, dimensions=3)
    waypoints = [[1, 0, 0], [2, -1, 1], [2, 0, 1], [1, 1, 1], [0, 0, 1], [0, 2, 2]]
    timestamps = [0, 1, 2, 3, 4, 5]
    solver.solve(waypoints, timestamps)
    solver.show_path()
