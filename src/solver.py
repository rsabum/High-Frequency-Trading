import numpy as np
from tqdm import tqdm
from .simulation import MarketState


class ValueFunction(object):
    def __init__(self, t_grid, q_grid, V):
        """
        Initializes the SolverOutput class with the specified parameters.

        Parameters:
        -----------
        q_grid : list
            A list representing the q grid.
        t_grid : list
            A list representing the t grid.
        V : np.ndarray
            A numpy array representing the optimal 
            value function at each q and t value.
        U : dict
            A dictionary representing the optimal policy 
            at each q and t value.
        """

        self.q_grid = q_grid
        self.t_grid = t_grid

        self.t_lookup = {t: i for i, t in enumerate(t_grid)}
        self.q_lookup = {q: i for i, q in enumerate(q_grid)}

        self.V = V
    
    def __call__(self, state: MarketState) -> float:
        t = state.timestamp
        q = state.inventory

        t_idx = self.t_lookup[t]
        q_idx = self.q_lookup[q]

        return self.V[t_idx, q_idx]


class Policy(object):
    def __init__(self, t_grid, q_grid, U):
        """
        Initializes the SolverOutput class with the specified parameters.

        Parameters:
        -----------
        q_grid : list
            A list representing the q grid.
        t_grid : list
            A list representing the t grid.
        V : np.ndarray
            A numpy array representing the optimal 
            value function at each q and t value.
        U : dict
            A dictionary representing the optimal policy 
            at each q and t value.
        """

        self.q_grid = q_grid
        self.t_grid = t_grid

        self.t_lookup = {t: i for i, t in enumerate(t_grid)}
        self.q_lookup = {q: i for i, q in enumerate(q_grid)}

        self.U = U
    
    def __call__(self, state: MarketState) -> tuple:
        t = state.timestamp
        q = state.inventory

        t_idx = self.t_lookup[t]
        q_idx = self.q_lookup[q]

        return self.U[t_idx, q_idx]


class HJB_QVI_Solver(object):
    
    @staticmethod
    def solve(
        T, N, q_min, q_max,lambda_bid, lambda_ask, 
        kappa_bid, kappa_ask, rebate,cost,phi, alpha
    ):
        """
        Solves the optimal value function and policy 
        function by solving the HJB-QVI.

        Parameters:
        -----------
        T : float
            The terminal time.
        N : int
            The number of time steps.
        q_min : int
            The minimum inventory level.
        q_max : int
            The maximum inventory level.
        lambda_bid : float
            The bid side order flow intensity.
        lambda_ask : float
            The ask side order flow intensity.
        kappa_bid : float
            The bid side order flow decay.
        kappa_ask : float
            The ask side order flow decay.
        cost : float
            The cost of sending a market order.
        rebate : float
            The rebate for providing liquidity.
        phi : float
            The running inventory penalty parameter.
        alpha : float
            The terminal inventory penalty parameter.

        Returns:
        --------
        V : ValueFunction
            The optimal value function.
        U : Policy
            The optimal policy.
        """         

        # Compute the time step size
        dt = T / N

        # Initialize time and inventory grids
        t_grid = np.linspace(0, T, N + 1)
        q_grid = np.arange(q_min, q_max + 1)

        # Initialize the value function and policy
        V = np.zeros((len(t_grid), len(q_grid)))
        U = {}

        # Set the terminal condition: V(T, q) = -alpha * q^2
        V[-1, :] = -alpha * q_grid ** 2

        # Solve the control problem by using 
        # a backwards euler finite difference scheme to
        # solve the HJB equation and value iteration to solve the QVI

        # Iterate backwards in time
        for i in tqdm(range(N, 0, -1), desc="Solving HJB-QVI"):
            # initialize error to some large value
            error = 1e9

            # solve for the value function at time t_{i - 1}
            while error > 1e-6:
                # initialize the value function at time t_{i - 1}
                V_prime = np.zeros(len(q_grid))

                for j, q_j in enumerate(q_grid):
                    # Compute the analytically optimal bid and ask depths
                    B = max(
                        0, 1/kappa_bid - rebate - 
                        (V[i, j + 1] - V[i, j])) if q_j < q_max else None
                    
                    A = max(
                        0, 1/kappa_ask - rebate - 
                        (V[i, j - 1] - V[i, j])) if q_j > q_min else None
                    
                    # Compute the value for each possible action
                    # (market buy, market sell, market make)
                    V_mb = V[i - 1, j + 1] - cost if q_j < q_max else -np.inf
                    V_ms = V[i - 1, j - 1] - cost if q_j > q_min else -np.inf
                    V_mm = V[i, j] + dt * (
                        (lambda_bid * np.exp(-kappa_bid * B) * 
                        (B + rebate + V[i, j + 1] - V[i, j]) if B else 0) + 
                        (lambda_ask * np.exp(-kappa_ask * A) * 
                        (A + rebate + V[i, j - 1] - V[i, j]) if A else 0) - 
                        phi * q_j ** 2
                    )

                    # Find the optimal action
                    k = np.argmax([V_mb, V_ms, V_mm])

                    # Update the value function and policy
                    if k == 0:
                        V_prime[j] = V_mb
                        U[(i - 1, j)] = ("market_buy", None, None)
                    elif k == 1:
                        V_prime[j] = V_ms
                        U[(i - 1, j)] = ("market_sell", None, None)
                    else:
                        V_prime[j] = V_mm
                        U[(i - 1, j)] = ("market_make", B, A)

                # Compute the error and update the value function
                error = np.linalg.norm(V_prime - V[i - 1])
                V[i - 1] = V_prime
                    
        # Store the value function and policy
        V = ValueFunction(t_grid, q_grid, V)
        U = Policy(t_grid, q_grid, U)

        return V, U
        