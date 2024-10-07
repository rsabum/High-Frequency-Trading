import numpy as np

class MarketState(object):

    def __init__(
        self,
        timestamp: float,
        position: int,
        cash: float,
        mid_price: float
    ):

        self.timestamp = timestamp
        self.position = position
        self.cash = cash
        self.mid_price = mid_price

class MarketSimulation():
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

    lambda_bid : float
        the intensity of the Poisson process dictating the arrivals of buy market orders
    lambda_ask : float
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
        T: float, 
        N: int, 
        q_min: int, 
        q_max: int,

        mu: float,
        sigma: float,
        S_0: float,
        ds: float,

        lambda_bid: float, 
        lambda_ask: float, 
        kappa_bid: float, 
        kappa_ask: float, 

        rebate: float,
        cost: float,

        debug: bool=False,
    ):

        self.T = T
        self.N = N
        self.dt = T / N
        self.t_grid = np.linspace(0, self.T, self.N + 1)

        self.q_min = q_min
        self.q_max = q_max

        self.mu = mu
        self.sigma = sigma
        self.S_0 = S_0
        self.ds = ds

        self.lambda_bid = lambda_bid
        self.lambda_ask = lambda_ask
        self.kappa_bid = kappa_bid
        self.kappa_ask = kappa_ask

        self.rebate = rebate
        self.cost = cost

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
            (self.mu - 0.5 * self.sigma ** 2) * self.dt + 
            self.sigma * dW
        )
        
        # Round midprice to nearest tick
        self.S_t = self.round_to_tick(self.S_t)
    
    
    def state(self) -> MarketState:
        """
        Returns the observation space

        Parameters
        ----------
        None

        Returns
        -------
        observation : MarketState
            The current state of the environment
        """

        return MarketState(
            timestamp=self.t_grid[self.t_i],
            position=self.Q_t,
            cash=self.X_t,
            mid_price=self.S_t
        )

    def market_make(self, bid: float, ask: float) -> tuple[float, float]:
        if self.debug:
            print("Market Making...")
            print("\tBid:", round(self.S_t - bid, 2))
            print("\tAsk:", round(self.S_t + ask, 2))
            print("\tSpread:", bid + ask)

        # ----- SAMPLE NUMBER of EXECUTED ORDERS -----
        
        # Number of orders executed
        n_exec_MO_bid = np.random.poisson(self.lambda_bid * np.exp(-self.kappa_bid * bid) * self.dt)
        n_exec_MO_ask = np.random.poisson(self.lambda_ask * np.exp(-self.kappa_ask * ask) * self.dt)
        if self.debug:
            print("\tBid Side Orders:", n_exec_MO_bid)
            print("\tAsk Side Orders:", n_exec_MO_ask)

        # Change in inventory and cash processes
        dX, dQ = 0, 0

        # match executed buy orders with executed sell orders
        dX += (bid + ask + 2 * self.rebate) * min(n_exec_MO_bid, n_exec_MO_ask)
        dQ = n_exec_MO_bid - n_exec_MO_ask
        
        if dQ > 0:
            # fill orders up to maximum inventory
            dX -= (self.S_t - bid) * (min(dQ, self.q_max - self.Q_t))
            dQ = min(dQ, self.q_max - self.Q_t)

        elif dQ < 0:
            # fill orders down to minimum inventory
            dX += (self.S_t + ask) * abs(max(dQ, self.q_min - self.Q_t))
            dQ = max(dQ, self.q_min - self.Q_t)

        return dX, dQ


    def market_take(self, buy: bool) -> tuple[float, float]:
        if self.debug:
            print("Market Taking...")
            if buy:
                print(f"\tBuying {1} share at {self.S_t}")
            else:
                print(f"\tSelling {1} share at {self.S_t}")
                
        # Change in inventory and cash processes
        dX, dQ = 0, 0

        # match executed buy orders with executed sell orders
        dX = (-self.S_t if buy else self.S_t) - self.cost
        dQ = 1 if buy else -1

        return dX, dQ


    def step(self, action: tuple) -> tuple[MarketState, float, bool]:
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

        choice, bid, ask = action
        
        if self.debug:
            print("-" * 30)
            print("t =", self.t_grid[self.t_i])
            print("Q_t =", round(self.Q_t, 2))
            print("X_t =", round(self.X_t, 2))
            print("S_t =", round(self.S_t, 2))

        if choice == "market_make":
            # Increment time only when we market make
            self.t_i += 1
            self.update_price()
            dX, dQ = self.market_make(bid, ask)
        
        elif choice == "market_buy":
            dX, dQ = self.market_take(True)
        
        elif choice == "market_sell":
            dX, dQ = self.market_take(False)

        if self.debug:
            print(f"Profit Earned: {round(dX, 2)}")
            print(f"Net Inventory Change: {dQ}")

        self.X_t += dX
        self.Q_t += dQ

        wealth = self.X_t + self.S_t * self.Q_t

        if self.debug:
            print("-" * 30)

        return self.state(), wealth, self.t_i == self.N, self.t_i
        

    def reset(self) -> MarketState:
        """
        Resets the environment

        Parameters
        ----------
        None

        Returns
        -------
        observation : MarketState
            The initial observation of the environment
        """

        self.t_i = 0
        self.Q_t = 0
        self.X_t = 0
        self.S_t = self.S_0

        return self.state()