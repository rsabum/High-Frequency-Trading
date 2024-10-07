import numpy as np
from tqdm import tqdm
from .simulation import MarketState

class KalmanFilter(object):
    pass

class ValueFunction(object):
    def __init__(self, t_grid, q_grid, V, PI):
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

        self.q_grid = q_grid
        self.t_grid = t_grid

        self.t_lookup = {t: i for i, t in enumerate(t_grid)}
        self.q_lookup = {q: i for i, q in enumerate(q_grid)}

        self.V = V
        self.PI = PI
    
    def get_value(self, t, q):
        """
        Returns the policy at the specified time and inventory level.

        Parameters:
        -----------
        t : float
            The time at which to evaluate the policy.
        q : float
            The inventory level at which to evaluate the policy.

        Returns:
        --------
        tuple
            The policy at the specified time and inventory level.
        """

        # Find the closest time step to the specified time
        t_idx = self.t_lookup[t]
        q_idx = self.q_lookup[q]

        return self.V[t_idx, q_idx]
    
    def get_policy(self, t, q):
        """
        Returns the policy at the specified time and inventory level.

        Parameters:
        -----------
        t : float
            The time at which to evaluate the policy.
        q : float
            The inventory level at which to evaluate the policy.

        Returns:
        --------
        tuple
            The policy at the specified time and inventory level.
        """

        # Find the closest time step to the specified time
        t_idx = self.t_lookup[t]
        q_idx = self.q_lookup[q]

        return self.PI[(t_idx, q_idx)]


class MarketMaker(object):
    def __init__(
        self, 
        T, 
        N, 
        q_min, 
        q_max,
        lambda_bid, 
        lambda_ask, 
        kappa_bid, 
        kappa_ask, 
        rebate,
        cost,
        phi, 
        alpha
    ):
        """
        Initializes the MarketMaker class with the specified parameters.

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
        None
        """    
        
        self.T = T
        self.N = N
        self.q_min = q_min
        self.q_max = q_max
        self.lambda_bid = lambda_bid
        self.lambda_ask = lambda_ask
        self.kappa_bid = kappa_bid
        self.kappa_ask = kappa_ask
        self.rebate = rebate
        self.cost = cost
        self.phi = phi
        self.alpha = alpha

        self.V = None

    def solve_hjb_qvi(self):
        """
        Solves the optimal value function and policy 
        function using the HJB-QVI.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """         

        # Compute the time step size
        dt = self.T / self.N

        # Initialize time and inventory grids
        t_grid = np.linspace(0, self.T, self.N + 1)
        q_grid = np.arange(self.q_min, self.q_max + 1)

        # Initialize the value function and policy
        V = np.zeros((len(t_grid), len(q_grid)))
        PI = {}

        # Set the terminal condition: V(T, q) = -alpha * q^2
        V[-1, :] = -self.alpha * q_grid ** 2

        # Solve the control problem by using 
        # a backwards euler finite difference scheme to
        # solve the HJB equation and value iteration to solve the QVI
        print("Solving HJB-QVI...")

        # Set the maximum error tolerance for value iteration
        max_error = 1e-9

        # Iterate backwards in time
        for i in tqdm(range(self.N, 0, -1)):
            # initialize error to some large value
            error = 1e9

            # solve for the value function at time t_{i - 1}
            while error > max_error:
                # initialize the value function at time t_{i - 1}
                V_prime = np.zeros(len(q_grid))

                for j, q_j in enumerate(q_grid):
                    # Compute the analytically optimal bid and ask depths
                    B = max(
                        0, 1/self.kappa_bid - self.rebate - 
                        (V[i, j + 1] - V[i, j])) if q_j < self.q_max else None
                    
                    A = max(
                        0, 1/self.kappa_ask - self.rebate - 
                        (V[i, j - 1] - V[i, j])) if q_j > self.q_min else None
                    
                    # Compute the value for each possible action
                    # (market buy, market sell, market make)
                    V_mb = V[i - 1, j + 1] - self.cost if q_j < self.q_max else -np.inf
                    V_ms = V[i - 1, j - 1] - self.cost if q_j > self.q_min else -np.inf
                    V_mm = V[i, j] + dt * (
                        (self.lambda_bid * np.exp(-self.kappa_bid * B) * 
                        (B + self.rebate + V[i, j + 1] - V[i, j]) if B else 0) + 
                        (self.lambda_ask * np.exp(-self.kappa_ask * A) * 
                        (A + self.rebate + V[i, j - 1] - V[i, j]) if A else 0) - 
                        self.phi * q_j ** 2
                    )

                    # Find the optimal action
                    k = np.argmax([V_mb, V_ms, V_mm])

                    # Update the value function and policy
                    if k == 0:
                        V_prime[j] = V_mb
                        PI[(i - 1, j)] = ("market_buy", None, None)
                    elif k == 1:
                        V_prime[j] = V_ms
                        PI[(i - 1, j)] = ("market_sell", None, None)
                    else:
                        V_prime[j] = V_mm
                        PI[(i - 1, j)] = ("market_make", B, A)

                # Compute the error and update the value function
                error = np.linalg.norm(V_prime - V[i - 1])
                V[i - 1] = V_prime
                    
        # Store the value function and policy
        self.V = ValueFunction(t_grid, q_grid, V, PI)

        print("HJB-QVI Solved!")
                    
    
    def run(self, state: MarketState) -> tuple:
        return self.V.get_policy(state.timestamp, state.position)