import numpy as np

class MarketState(object):

    def __init__(
        self,
        timestamp: float,
        inventory: int,
        cash: float,
        mid_price: float
    ):

        self.timestamp = timestamp
        self.inventory = inventory
        self.cash = cash
        self.mid_price = mid_price

class MarketSimulation():
    """
    Attributes
    ----------
    T : integer
        the number of periods to run the simulation
    N : int
        the number of timesteps per simulation
    q_min : int
        the minimum inventory level
    q_max : int
        the maximum inventory level
    X_0 : float
        the initial cash position of the agent
    S_0 : float
        Initial asset price.
    mu : float
        The drift of the asset.
    sigma : float
        The volatility of the asset.
    ds : float
        Asset price tick size.
    lambda_bid : float
        the intensity of the Poisson process dictating the arrivals 
        of market orders on the bid side
    lambda_ask : float
        the intensity of the Poisson process dictating the arrivals 
        of market orders on the ask side
    kappa_bid : float
        order flow decay parameter for the bid side
    kappa_ask : float
        order flow decay parameter for the ask side
    alpha : float
        the terminal inventory penalty parameter
    phi : float
        the running inventory penalty parameter
    rebate_rate : float
        rebate rate awarded for providing liquidity
    cost : float
        cost of sending a market order
    debug : bool
        whether or not information for debugging should be printed during simulation
    """

    def __init__(
        self, 
        T: float, 
        N: int, 
        q_min: int, 
        q_max: int,
        X_0: float,
        S_0: float,
        mu: float,
        sigma: float,
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

        self.X_0 = X_0
        self.S_0 = S_0
        self.mu = mu
        self.sigma = sigma
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
        
        # Generate Weiner Process increment
        dW = np.random.normal(scale=np.sqrt(self.dt))

        # Simulate the asset price process
        self.S_t = self.S_t + self.mu * self.dt + self.sigma * dW
        
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
            inventory=self.Q_t,
            cash=self.X_t,
            mid_price=self.S_t
        )

    def market_make(self, bid: float, ask: float) -> tuple[float, float]:
        """
        Simulates the market maker quoting a bid and ask depth
        
        Parameters
        ----------
        bid : float
            The bid depth
        ask : float
            The ask depth
            
        Returns
        -------
        dX : float
            The change in the cash process
        dQ : float
            The change in the inventory process
        """

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


    def market_take(self, buy: bool, n: int) -> tuple[float, float]:
        """
        Simulates the market maker sending a market order
        to buy or sell n shares

        Parameters
        ----------
        buy : bool
            Whether the market order is a buy or sell order
        n : int
            The number of shares to buy or sell

        Returns
        -------
        dX : float
            The change in the cash process 
        dQ : float
            The change in the inventory process
        """

        if self.debug:
            print("Market Taking...")
            if buy:
                print(f"\tBuying {n} shares at {self.S_t}")
            else:
                print(f"\tSelling {n} shares at {self.S_t}")
                
        # Change in inventory and cash processes
        dX, dQ = 0, 0

        # match executed buy orders with executed sell orders
        dX = n * (-self.S_t if buy else self.S_t) - n * self.cost
        dQ = n if buy else -n

        return dX, dQ


    def step(self, action: tuple) -> tuple[MarketState, float, bool]:
        """
        Takes a step in the environment based on the market maker quoting a bid and ask depth

        Parameters
        ----------
        action : tuple
            The action tuple containing the market maker's choice
            of action and the bid and ask depths/market order size

        Returns
        -------
        observation : ndarray
            The observation of the next state

        wealth : float
            The current wealth of the agent

        terminal : bool
            Whether the episode has terminated
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
            dX, dQ = self.market_take(True, 1)
        
        elif choice == "market_sell":
            dX, dQ = self.market_take(False, 1)

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
        self.X_t = self.X_0
        self.S_t = self.S_0

        return self.state()
