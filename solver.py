from functools import cache
import numpy as np
from tqdm import tqdm


class SolverOutput:
    def __init__(self, s_grid, q_grid, t_grid, V, PI):
        """
        Initializes the SolverOutput class with the specified parameters.

        Parameters:
        -----------
        q_grid : list
            A list representing the q grid.
        t_grid : list
            A list representing the t grid.
        h : np.ndarray
            A numpy array representing the optimal 
            value function at each q and t value.
        """

        self.s_grid = s_grid
        self.q_grid = q_grid
        self.t_grid = t_grid
        self.V = V
        self.PI = PI

    

class Finite_Difference_Solver:

    @staticmethod
    def solve(T, mu, theta, sigma, q_min, q_max, n_t, n_q, n_s, phi):
        """
        Solves the optimal order size for a long only 
        trading algorithm using backward Euler finite 
        difference scheme.

        Parameters:
        -----------
        T : float
            The number of time periods.
        mu : float
            The drift of the price process.
        theta : float
            The speed of mean reversion.
        sigma : float
            The volatility of the price process.
        q_min : int
            The minimum inventory level.
        q_max : int
            The maximum inventory level.
        phi : float
            The transaction costs.
        n_t : int
            The number of time steps.
        n_q : int
            The number of inventory levels.
        n_s : int
            The number of price levels

        Returns:
        --------
        SolverOutput
            The optimal value function.
        """         


        # time grid and time step size
        dt = T / n_t
        t_grid = np.zeros(n_t + 1)
        t_grid[-1] = T  # Set the last time step to terminal time

        # Calculate the number of possible inventory levels (q_grid)
        dq = (q_max - q_min) / n_q
        q_grid = np.arange(q_min, q_max + 1, dq)  # Inventory grid

        # price grid (considering 3 standard deviations around the mean)
        s_min = np.floor(mu - 3 * sigma / np.sqrt(2 * theta))
        s_max = np.ceil(mu + 3 * sigma / np.sqrt(2 * theta))
        ds = (s_max - s_min) / n_s
        s_grid = np.arange(s_min, s_max + 1, ds)  # Price grid

        # Initialize the value function V
        V = np.ones((len(s_grid), len(q_grid), len(t_grid))) * -np.inf
        # Set the boundary conditions
        V[:, 0, :] = 0; V[:, -1, :] = 0
        V[0, :, :] = 0; V[-1, :, :] = 0

        # Set the terminal condition: h(s, q, T) = q * s
        V[:, :, -1] = np.outer(s_grid, q_grid)

        # Initialize policy function PI
        PI = np.zeros((len(s_grid), len(q_grid), len(t_grid)))
        

        for n in tqdm(range(n_t)[::-1]):  # Iterate backward over time steps
            # Update the current time step
            t_grid[n] = t_grid[n + 1] - dt

            # Iterate over all price and inventory levels s, q and
            # calculate the value function at the current time step
            # using the backward Euler finite difference scheme.
            for i, s_i in enumerate(s_grid):
                for j, q_j in enumerate(q_grid):
                    # Skip the boundary points
                    if i == 0 or i == len(s_grid) - 1:
                        continue

                    if j == 0 or j == len(q_grid) - 1:
                        continue
                    
                    for q_prime in q_grid:
                        u = q_prime - q_j
                        J_u = V[i, j, n + 1] + dt * (
                            theta * (mu - s_i) * (V[i + 1, j, n + 1] - V[i - 1, j, n + 1]) / (2 * ds) +
                            0.5 * sigma ** 2 * (V[i + 1, j, n + 1] - 2 * V[i, j, n + 1] + V[i - 1, j, n + 1]) / ds ** 2 +
                            (q_j + u) * (V[i, j + 1, n + 1] - V[i, j - 1, n + 1]) / (2 * dq) - 
                            phi * abs(u) - 
                            s_i * (u)
                        )

                        if J_u > V[i, j, n]:
                            V[i, j, n] = J_u
                            PI[i, j, n] = u
                    
                    
        # Return the solution in the form of a SolverOutput object
        return SolverOutput(s_grid, q_grid, t_grid, V, PI)

            
    
