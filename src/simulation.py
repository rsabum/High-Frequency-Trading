import numpy as np

class MarketState(object):

    def __init__(
        self,
        timestamp: float,
        mid_price: float,
        position: int,
        cash: float
    ):

        self.timestamp = timestamp
        self.mid_price = mid_price
        self.position = position
        self.cash = cash

class MaketSimulation():
    """
    SimpleEnv is an object that is able to simulate the simple probabilistic environment.

    Attributes
    ----------
    T : integer
        the number of periods to run the simulation
    M : int
        the number of timesteps per simulation
    dt : integer
        the size of the time steps

    sigma : float
        The volatility of the asset.
    S0 : float
        Initial asset price.
    ds : float
        Asset price tick size.

    lambda_buy : float
        the intensity of the Poisson process dictating the arrivals of buy market orders
    lambda_sell : float
        the intensity of the Poisson process dictating the arrivals of sell market orders
    kappa_bid : float
        decay parameter for the execution probability of bid quotes
    kappa_ask : float
        decay parameter for the execution probability of ask quotes
    
    alpha : float
        the terminal inventory penalty parameter
    phi : float
        the running inventory penalty parameter

    rebate_rate : float
        rebate rate awarded for providing liquidity

    debug : bool
        whether or not information for debugging should be printed during simulation
    """

    def __init__(
        self,

        T=1,
        N=10,

        mu=0.1,
        sigma=3e-2,
        S_0=100,
        ds=0.01,

        lambda_buy=10,
        lambda_sell=10,
        kappa_bid=100,
        kappa_ask=100,
        
        phi=1e-5,
        alpha=1e-3,

        rebate_rate=0.0025,

        debug=False,
    ):

        self.T = T
        self.N = N
        self.dt = T / N

        self.mu = mu
        self.sigma = sigma
        self.S_0 = S_0
        self.ds = ds

        self.lambda_buy = lambda_buy
        self.lambda_sell = lambda_sell
        self.kappa_bid = kappa_bid
        self.kappa_ask = kappa_ask

        self.phi = phi
        self.alpha = alpha

        self.rebate_rate = rebate_rate

        self.debug = debug

        # Reset the environment
        self.reset()


    def round_to_tick(self, price: float) -> float:
        """
        Round a price to the closest tick

        Parameters
        ----------
        price : float
            the input price

        Returns
        -------
        price : float
            the rounded price
        """

        return np.round(price / self.ds, decimals=0) * self.ds
        

    def update_price(self):
        """
        Updates the mid price once and rounds it to the nearest tick

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        
        # Generate Brownian motion increments
        dW = np.random.normal(scale=np.sqrt(self.dt))

        # Simulate the asset price process
        self.S_t = self.S_t * np.exp(
            np.sqrt(self.sigma) * dW
        )
        
        # Round midprice to nearest tick
        self.S_t = self.round_to_tick(self.S_t)
    
    
    def state(self) -> np.ndarray:
        """
        Returns the observation space

        Parameters
        ----------
        None

        Returns
        -------
        observation : np.ndarray
            the observation in terms of (cash, mid price, inventory, time) = (X_t, S_t, Q_t, t)
        """

        return np.array([self.X_t, self.S_t, self.Q_t, self.t], dtype=np.float64)


    def step(self, bid: float, ask: float) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Takes a step in the environment based on the market maker quoting a bid and ask depth

        Parameters
        ----------
        action : np array
            The bid and ask depth quotes

        Returns
        -------
        observation : ndarray
            The observation of the next state

        reward : float
            The immediate reward of the resulting observation

        terminal : bool
            Whether the episode has terminated

        truncated : bool
            Whether the episode has been truncated

        info : dict
            Additional information about the environment 
        """

        
        if self.debug:
            print("-" * 30)
            print("t =", self.t)
            print("X_t =", round(self.X_t, 2))
            print("Q_t =", round(self.Q_t, 2))
            print("S_t =", round(self.S_t, 2))
            print()
            print("Bid:", round(self.S_t - bid, 2))
            print("Ask:", round(self.S_t + ask, 2))
            print("Spread:", bid + ask)

        # ----- SAMPLE NUMBER of EXECUTED ORDERS -----
        
        # Number of market orders that arrive
        n_MO_buy = np.random.poisson(self.lambda_buy * self.dt)
        n_MO_sell = np.random.poisson(self.lambda_sell * self.dt)

        # Order Execution Probability
        p_MO_buy = np.exp(-self.kappa_ask * ask)
        p_MO_sell = np.exp(-self.kappa_bid * bid)

        # Number of orders executed
        n_exec_MO_buy = np.random.binomial(n_MO_buy, p_MO_buy)
        n_exec_MO_sell = np.random.binomial(n_MO_sell, p_MO_sell)

        # Change in inventory and cash processes
        dQ = n_exec_MO_sell - n_exec_MO_buy
        dX = (ask * n_exec_MO_buy) + (bid * n_exec_MO_sell) + self.rebate_rate * (n_exec_MO_sell + n_exec_MO_buy)


        if self.debug:
            print(f"Incomming Buy Market Orders: {n_MO_buy}")
            print(f"Incomming Sell Market Orders: {n_MO_sell}")

            print(f"Buy Order Execution Probability ~ {round(p_MO_buy, 2)}")
            print(f"Sell Order Execution Probability ~ {round(p_MO_sell, 2)}")

            print(f"Executed Buy Orders: {n_exec_MO_buy}")
            print(f"Executed Sell Orders: {n_exec_MO_sell}")

            print(f"Profit Earned: {round(dX, 2)}")

        self.X_t += dX
        self.Q_t += dQ

        self.t += 1
        self.update_price()

        # If we're at the final time step the MM must liquidate its inventory
        if self.t >= self.N:
            reward = self.X_t + self.Q_t * (self.S_t - (self.alpha * self.Q_t))
            terminal = True

            self.X_t = self.X_t + (self.Q_t * self.S_t)
            self.Q_t = 0

        else:
            reward = -self.phi * (self.Q_t ** 2)
            terminal = False


        truncated = False
        info = {}

        if self.debug:
            print("\nX_t =", round(self.X_t, 2))
            print("Q_t = ", self.Q_t)
            print("-" * 30)

        return (
            self.state(), 
            reward, 
            terminal, 
            truncated, 
            info
        )


    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        """
        Resets the environment

        Parameters
        ----------
        seed : int
            The seed for the random number generator

        options : dict
            Additional options for the environment

        Returns
        -------
        observation : numpy.ndarray
            The initial observation of the environment

        info : dict
            Additional information about the environment
        """

        self.t = 0

        self.S_t = self.S_0

        self.Q_t = 0
        self.X_t = 0

        return self.state(), {}