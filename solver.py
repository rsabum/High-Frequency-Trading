from functools import cache
import numpy as np
from tqdm import tqdm
from itertools import product
from scipy.stats import binom, poisson


class Market_Parameters:
    def __init__(self, lambda_buy, lambda_sell, kappa_buy, kappa_sell,
                 phi, alpha, q_min, q_max, T):
        
        if not isinstance(lambda_buy, (float, int, np.int32, np.int64)):
            raise TypeError(f'lambda_buy has to be type of <float> or <int>, not {type(lambda_buy)}')

        if not isinstance(lambda_sell, (float, int, np.int32, np.int64)):
            raise TypeError(f'lambda_sell has to be type of <float> or <int>, not {type(lambda_buy)}')       

        if not isinstance(phi, (float, int, np.int32, np.int64)):
            raise TypeError('phi has to be type of <float> or <int>')

        if not isinstance(alpha, (float, int, np.int32, np.int64)):
            raise TypeError('alpha has to be type of <float> or <int>')            

        if not isinstance(q_min, (int, np.int32, np.int64)):
            raise TypeError('q_min has to be type of <int>')  
            
        if not isinstance(q_max, (int, np.int32, np.int64)):
            raise TypeError('q_max has to be type of <int>')  
        
        if q_max <= q_min:
            raise ValueError('q_max has to be larger than q_min!')

        if not isinstance(T, (float, int, np.int32, np.int64)):
            raise TypeError('T has to be type of <int>')

        self.lambda_buy = lambda_buy      # Order flow at be bid
        self.lambda_sell = lambda_sell    # Order flow at the ask
        self.kappa_buy = kappa_buy        # Order flow decay at be bid
        self.kappa_sell = kappa_sell      # Order flow decay at the ask
        
        self.phi = phi        # running inventory penalty
        self.alpha = alpha    # terminal inventory penalty
        
        self.q_min = q_min    # largest allowed short position
        self.q_max = q_max    # largest allowed long position

        self.T = T


class SolverOutput:
    def __init__(self, q_grid, t_grid, h):
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

        self.h = h
        self.q_grid = q_grid
        self.t_grid = t_grid

    

class Solver:

    @cache
    @staticmethod
    def poisson_range(mu):
        """
        Uses binary search to find the range of probable values
        of a Poisson random variable with rate parameter mu.

        Parameters:
        -----------
        mu : float
            The rate parameter of the Poisson distribution.

        Returns:
        --------
        lb : int
            The lower bound of the probable values.
        ub : int
            The upper bound of the probable values.
        """

        # Initialize the lower bound (lb) to 0.
        lb = 0

        # Initialize the step size (j) for binary search to the rate parameter mu.
        j = mu

        # Binary search for the lower bound of the Poisson distribution.
        # The search continues while j (step size) is greater than or equal to 1.
        while j >= 1:
            # Increase lb by j until the cumulative distribution function (CDF)
            # of the Poisson distribution at (lb + j) exceeds a small threshold (0.001).
            # This threshold ensures that the values below this bound are very unlikely.
            while poisson.cdf(lb + j, mu) <= 0.001:
                lb = lb + j
            # Halve the step size for the next iteration of binary search.
            j = j // 2

        # Initialize the upper bound (ub) to the rate parameter mu.
        ub = mu

        # Reset the step size (j) to the rate parameter mu.
        j = mu

        # Binary search for the upper bound of the Poisson distribution.
        # The search continues while j (step size) is greater than or equal to 1.
        while j >= 1:
            # Increase ub by j until the CDF of the Poisson distribution at (ub + j)
            # exceeds a large threshold (0.999). This threshold ensures that the values
            # above this bound are very unlikely.
            while poisson.cdf(ub + j, mu) <= 0.999:
                ub = ub + j
            # Halve the step size for the next iteration of binary search.
            j = j // 2

        # Return the lower bound (lb) and the upper bound (ub), with the upper bound
        # incremented by 1 to make it inclusive.
        return int(lb), int(ub) + 1


    @cache
    @staticmethod
    def P_N(n, delta, mu, kappa):
        """
        Calculate the probability of n orders being executed
        at a given market depth (delta), given the order flow (mu) 
        and order flow decay (kappa).
        
        Parameters:
        -----------
        n : int
            The number of orders to be executed.
        delta : float
            The market depth.
        mu : float
            The order flow.
        kappa : float
            The order flow decay.

        Returns:
        --------
        prob : float
            The probability of n orders being executed.
        """

        prob = 0  # Initialize the probability to 0.

        # Calculate the probability p of each individual order being executed
        # as a function of the market depth (delta) and order flow decay (kappa).
        p = np.exp(-kappa * delta)
        
        # Determine the minimum and maximum range for the Poisson-distributed 
        # random variable m, which represents the number of potential orders.
        m_min, m_max = Solver.poisson_range(mu)
        
        # Sum over all possible values of m, where m represents the number of potential orders.
        # The range of m is bounded by the minimum and maximum values determined by the Poisson distribution.
        for m in range(min(n, m_min), m_max):
            # Calculate the probability of exactly n orders being executed out of m potential orders,
            # where each order has a success probability of p. This is modeled using the binomial distribution.
            binomial_prob = binom.pmf(n, m, p)
            
            # Calculate the probability of observing m potential orders, given that the order flow follows a 
            # Poisson distribution with mean mu.
            poisson_prob = poisson.pmf(m, mu)
            
            # Add the product of these probabilities to the total probability.
            prob += binomial_prob * poisson_prob

        return prob  # Return the total probability of executing exactly n orders.



    @cache
    @staticmethod
    def P_dQ(dQ, delta_bid, delta_ask, lambda_buy, lambda_sell, kappa_buy, kappa_sell):
        """
        Calculate the probability of observing an inventory change equal 
        to dQ given the bid and ask depths, order flow, and order flow decay.

        Parameters:
        -----------
        dQ : int
            The inventory change.
        delta_bid : float
            The bid depth.
        delta_ask : float
            The ask depth.
        lambda_buy : float
            The order flow at the bid.
        lambda_sell : float
            The order flow at the ask.
        kappa_buy : float
            The order flow decay at the bid.
        kappa_sell : float 
            The order flow decay at the ask.

        Returns:
        --------
        prob : float
            The probability of observing an inventory change equal to dQ.
        """

        # Case 1: Positive inventory change (dQ > 0) implies a net purchase of assets.
        if dQ > 0:
            return sum([
                # Probability of executing (dQ + n) buy orders and n sell orders
                Solver.P_N(dQ + n, delta_bid, lambda_buy, kappa_buy) * 
                Solver.P_N(n, delta_ask, lambda_sell, kappa_sell)
                for n in Solver.poisson_range(lambda_sell)
            ]) 

        # Case 2: No inventory change (dQ == 0) implies a balanced order book (buy = sell).
        if dQ == 0:
            return sum([
                # Probability of executing n buy and n sell orders
                Solver.P_N(n, delta_ask, lambda_sell, kappa_sell) * 
                Solver.P_N(n, delta_bid, lambda_buy, kappa_buy)
                for n in Solver.poisson_range(lambda_sell)
            ])

        # Case 3: Negative inventory change (dQ < 0) implies a net sale of assets.
        if dQ < 0:
            return sum([
                # Probability of executing abs(dQ) + n sell orders and n buy orders
                Solver.P_N(abs(dQ) + n, delta_ask, lambda_sell, kappa_sell) * 
                Solver.P_N(n, delta_bid, lambda_buy, kappa_buy)
                for n in Solver.poisson_range(lambda_buy)
            ])

        

    @cache
    @staticmethod
    def E_dX(dQ, delta_bid, delta_ask, lambda_buy, lambda_sell, kappa_buy, kappa_sell):
        """
        Calculate the expected trading profit given an inventory change equal to dQ
        and the bid and ask depths, order flow, and order flow decay.

        Parameters:
        -----------
        dQ : int
            The inventory change.
        delta_bid : float
            The bid depth.
        delta_ask : float
            The ask depth.
        lambda_buy : float
            The order flow at the bid.
        lambda_sell : float
            The order flow at the ask.
        kappa_buy : float
            The order flow decay at the bid.
        kappa_sell : float
            The order flow decay at the ask.

        Returns:
        --------
        e_dX : float
            The expected trading profit given an inventory change equal to dQ.
        """

        # Case 1: Positive inventory change (dQ > 0) implies a net purchase of assets.
        if dQ > 0:
            return sum([
                # Probability of executing (dQ + n) buy orders and n sell orders
                Solver.P_N(dQ + n, delta_bid, lambda_buy, kappa_buy) * 
                Solver.P_N(n, delta_ask, lambda_sell, kappa_sell) *
                # Trading profit: revenue from selling n units minus cost of buying (dQ + n) units
                ((1 + delta_ask) * n - (1 - delta_bid) * (dQ + n))
                for n in Solver.poisson_range(lambda_sell)
            ]) 
        
        # Case 2: No inventory change (dQ == 0) implies a balanced order book (buy = sell).
        if dQ == 0:
            return sum([
                # Probability of executing n buy and n sell orders
                Solver.P_N(n, delta_ask, lambda_sell, kappa_sell) * 
                Solver.P_N(n, delta_bid, lambda_buy, kappa_buy) *
                # Trading profit: revenue from selling n units minus cost of buying n units
                ((1 + delta_ask) * n - (1 - delta_bid) * n)
                for n in Solver.poisson_range(lambda_sell)
            ])

        # Case 3: Negative inventory change (dQ < 0) implies a net sale of assets.
        if dQ < 0:
            return sum([
                # Probability of executing abs(dQ) + n sell orders and n buy orders
                Solver.P_N(abs(dQ) + n, delta_ask, lambda_sell, kappa_sell) * 
                Solver.P_N(n, delta_bid, lambda_buy, kappa_buy) *
                # Trading profit: revenue from selling (abs(dQ) + n) units minus cost of buying n units
                ((1 + delta_ask) * (abs(dQ) + n) - (1 - delta_bid) * n)
                for n in Solver.poisson_range(lambda_buy)
            ])



    @staticmethod
    def solve(params, N_steps):
        """
        Solves the optimal value function for the given parameters
        using backward stochastic dynamic programming.

        Parameters:
        -----------
        params : Market_Parameters
            The market parameters.
        N_steps : int
            The number of time steps.

        Returns:
        --------
        SolverOutput
            The optimal value function.
        """         

        # Calculate the number of possible inventory levels (q_grid) 
        n = params.q_max - params.q_min + 1 
        q_grid = np.arange(params.q_min, params.q_max + 1)  # Inventory grid

        # Terminal time and time step size
        T = params.T
        dt = T / N_steps
        t_grid = np.zeros(N_steps + 1)  # Time grid
        t_grid[-1] = T  # Set the last time step to terminal time

        # Initialize the value function h with -inf (representing the initial guess)
        h = np.ones((n, N_steps + 1)) * -np.inf
        # Set the terminal condition: h(q, T) = -alpha * q^2
        h[:, -1] = -params.alpha * q_grid ** 2
        
        # Backward Dynamic Programming loop to solve for h at each time step
        for t in tqdm(range(N_steps)[::-1]):  # Iterate backward over time steps
            # Update the current time step
            t_grid[t] = t_grid[t + 1] - dt

            # Iterate over all possible inventory levels q
            for q in range(params.q_min, params.q_max + 1):
                # Initialize the maximum expected value for h[q][t]
                for b, a in product(range(10), range(10)):  # Iterate over bid and ask depths
                    delta_bid, delta_ask = b / 100, a / 100  # Normalize depths to [0, 0.09]
                    
                    e_V = 0  # Expected value initialization

                    # Calculate the expected value by summing over all possible next inventory levels
                    for q_next in range(params.q_min, params.q_max + 1):
                        dQ = q_next - q  # Change in inventory

                        # Calculate the probability of observing this change in inventory
                        prob = Solver.P_dQ(
                            dQ, delta_bid, delta_ask, 
                            dt * params.lambda_buy, dt * params.lambda_sell, 
                            params.kappa_buy, params.kappa_sell
                        )

                        # Calculate the expected trading profit given this change in inventory
                        e_dX = Solver.E_dX(
                            dQ, delta_bid, delta_ask, 
                            dt * params.lambda_buy, dt * params.lambda_sell, 
                            params.kappa_buy, params.kappa_sell
                        )

                        # Add the expected value for this transition to the running sum
                        e_V += prob * (e_dX - params.phi * (q + dQ) ** 2 + h[q + dQ - params.q_min][t + 1])

                    # Update the value function with the maximum of the current expected value and previous one
                    h[q][t] = max(e_V, h[q][t])
        
        # Return the solution in the form of a SolverOutput object
        return SolverOutput(q_grid, t_grid, h)

            
    
