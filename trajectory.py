from cvxopt import matrix
from cvxopt.solvers import qp
import numpy as np
import matplotlib.pyplot as plt
import sympy
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class Solver:
    def __init__(self, d, r, q, dimensions=2):
        # degree of polynomials
        self.d = d
        # optimize r-th derivative
        self.r = r
        # ensure continuity of up to q-th derivative
        self.q = q
        # dimensions (2 or 3)
        self.dimensions = dimensions

        self.waypoints = None
        self.timestamps = None
        self.result = None

    def solve(self, waypoints, timestamps):
        waypoints = np.array(waypoints)
        self.waypoints = waypoints
        timestamps = np.array(timestamps)
        self.timestamps = timestamps
        n_segments = len(waypoints) - 1
        result = np.zeros((self.dimensions, n_segments, self.d + 1))
        P = np.zeros((n_segments * (self.d + 1), n_segments * (self.d + 1)))
        q = matrix(np.zeros(n_segments * (self.d + 1), dtype=np.float64))
        for d in range(self.dimensions):
            A = np.zeros(
                (
                    (n_segments - 1) * (self.q + 2) + 2 * (self.q + 1),
                    n_segments * (self.d + 1),
                ),
                dtype=np.float64,
            )
            b = np.zeros(
                (n_segments - 1) * (self.q + 2) + 2 * (self.q + 1), dtype=np.float64
            )
            A[: self.q + 1, : self.d + 1] = self.lin_matrix(0)
            b[0] = waypoints[0, d]
            for i in range(0, n_segments - 1):
                P_ = self.quad_matrix(timestamps[i + 1] - timestamps[i])
                P[
                    i * (self.d + 1) : (i + 1) * (self.d + 1),
                    i * (self.d + 1) : (i + 1) * (self.d + 1),
                ] = P_
                A0 = self.lin_matrix(0)
                AT = self.lin_matrix(timestamps[i + 1] - timestamps[i])
                # define constraints for waypoint locations
                A[
                    2 + i * (self.q + 2),
                    i * (self.d + 1) : (i + 1) * (self.d + 1),
                ] = AT[0]

                b[2 + i * (self.q + 2)] = waypoints[i + 1, d]
                # define constraints for equality of derivatives
                A[
                    2 + i * (self.q + 2) + 1 : 2 + (i + 1) * (self.q + 2),
                    i * (self.d + 1) : (i + 1) * (self.d + 1),
                ] = AT
                A[
                    2 + i * (self.q + 2) + 1 : 2 + (i + 1) * (self.q + 2),
                    (i + 1) * (self.d + 1) : (i + 2) * (self.d + 1),
                ] = -A0
            P_ = self.quad_matrix(timestamps[-1] - timestamps[-2])
            P[-(self.d + 1) :, -(self.d + 1) :] = P_
            A[-(self.q + 1) :, -(self.d + 1) :] = self.lin_matrix(
                timestamps[-1] - timestamps[-2]
            )
            b[-(self.q + 1)] = waypoints[-1, d]

            # remove linearly dependent rows
            _, inds = sympy.Matrix(A).T.rref()
            A = A[list(inds)]
            b = b[list(inds)]
            # for numerical stability add identity matrix times a small scalar for regularization
            P = P + 1e-5 * np.eye(n_segments * (self.d + 1))
            P = matrix(P)
            A = matrix(A)
            b = matrix(b.reshape(-1, 1))

            sol = qp(P, q, None, None, A, b)
            res_dim = np.array(sol["x"]).reshape((n_segments, self.d + 1))
            result[d, :, :] = res_dim
            self.result = result
        return result

    def quad_matrix(self, T):
        Q = np.zeros((self.d + 1, self.d + 1))
        for l in range(self.r, self.d + 1):
            for k in range(l, self.d + 1):
                q = (
                    np.prod([(l - m) * (k - m) for m in range(self.r)])
                    * T ** (l + k - 2 * self.r + 1)
                    / (l + k - 2 * self.r + 1)
                )
                Q[l, k] = q
                Q[k, l] = q
        return Q

    def lin_matrix(self, t):
        A = np.zeros((self.q + 1, self.d + 1))
        for i in range(self.q + 1):
            for j in range(i, self.d + 1):
                A[i, j] = np.prod(np.arange(j - i + 1, j + 1)) * t ** (j - i)
        return A

    def show_path(self):
        if self.dimensions == 2:
            self.show_path_2d()
        if self.dimensions == 3:
            self.show_path_3d()

    def show_path_2d(self):
        coeffs = self.result
        _, ax = plt.subplots()
        ax.plot(self.waypoints.T[0], self.waypoints.T[1], color="lime")
        ax.scatter(self.waypoints.T[0], self.waypoints.T[1], color="red")
        for p, label in zip(self.waypoints, self.timestamps):
            plt.text(p[0], p[1], str(label), fontsize=9, ha="right", va="bottom")
        x = []
        y = []
        v = []
        for i in range(coeffs.shape[1]):
            px = np.poly1d(coeffs[0, i][::-1])
            py = np.poly1d(coeffs[1, i][::-1])
            dpx = np.poly1d((coeffs[0, i] * np.arange(self.d + 1))[1:][::-1])
            dpy = np.poly1d((coeffs[1, i] * np.arange(self.d + 1))[1:][::-1])
            t = np.linspace(0, self.timestamps[i + 1] - self.timestamps[i], 100)
            x.append(px(t))
            y.append(py(t))
            dx = dpx(t)
            dy = dpy(t)
            v.append(np.sqrt((dx**2 + dy**2))[:-1])

        x = np.hstack(x)
        y = np.hstack(y)
        v = np.hstack(v)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap="plasma", array=v / np.max(v), linewidth=2)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label="velocity")
        plt.axis("equal")
        plt.show()

    def show_path_3d(self):
        coeffs = self.result
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            self.waypoints.T[0], self.waypoints.T[1], self.waypoints.T[2], color="lime"
        )
        ax.scatter(
            self.waypoints.T[0], self.waypoints.T[1], self.waypoints.T[2], color="red"
        )
        for p, label in zip(self.waypoints, self.timestamps):
            ax.text(p[0], p[1], p[2], str(label), fontsize=9, ha="right", va="bottom")
        x = []
        y = []
        z = []
        v = []
        for i in range(coeffs.shape[1]):
            px = np.poly1d(coeffs[0, i][::-1])
            py = np.poly1d(coeffs[1, i][::-1])
            pz = np.poly1d(coeffs[2, i][::-1])
            dpx = np.poly1d((coeffs[0, i] * np.arange(self.d + 1))[1:][::-1])
            dpy = np.poly1d((coeffs[1, i] * np.arange(self.d + 1))[1:][::-1])
            dpz = np.poly1d((coeffs[2, i] * np.arange(self.d + 1))[1:][::-1])
            t = np.linspace(0, self.timestamps[i + 1] - self.timestamps[i], 100)
            x.append(px(t))
            y.append(py(t))
            z.append(pz(t))
            dx = dpx(t)
            dy = dpy(t)
            dz = dpz(t)
            v.append(np.sqrt((dx**2 + dy**2 + dz**2))[:-1])

        x = np.hstack(x)
        y = np.hstack(y)
        z = np.hstack(z)
        v = np.hstack(v)
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, cmap="plasma", array=v / np.max(v), linewidth=2)
        ax.add_collection(lc)
        plt.colorbar(lc, ax=ax, label="velocity")
        plt.axis("equal")
        plt.show()
