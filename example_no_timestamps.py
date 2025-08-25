from trajectory import NoTimestampSolver

solver = NoTimestampSolver(d=5, r=2, q=2, dimensions=3)
waypoints = [[1, 0, 0], [2, -1, 1], [2, 0, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1.5]]
solver.solve(waypoints, 5)
solver.show_path()
