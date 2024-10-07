import numpy as np
import matplotlib.pyplot as plt
from src import MarketMaker, MarketSimulation

T = 60              # 5 minutes
N = T * 60          # one step per second

q_min = -50         # largest allowed short position
q_max = 50          # largest allowed long position

mu = 0.0001         # drift parameter
sigma = 0.01        # volatility parameter
S0 = 100            # initial price
ds = 0.01           # tick size

lambda_bid = 50     # Number of market order arrivals per minute on the bid
lambda_ask = 50     # Number of market order arrivals per minute on the ask
kappa_bid = 100     # Decay parameter for "fill rate" on the bid
kappa_ask = 100     # Decay parameter for "fill rate" on the ask

phi = 1e-6          # running inventory penalty parameter
alpha = 1e-4        # terminal inventory penalty parameter

rebate = 0.0025     # rebate rate for providing liquidity
cost = 0.005        # cost for taking liquidity

market = MarketSimulation(T, N, q_min, q_max, mu, sigma, S0, ds, lambda_bid, lambda_ask, kappa_bid, kappa_ask, rebate, cost)
agent = MarketMaker(T, N, q_min, q_max, lambda_bid, lambda_ask, kappa_bid, kappa_ask, rebate, cost, phi, alpha)
agent.solve_hjb_qvi()

W = np.zeros((10, N + 1))
for i in range(10):
    print("Running simulation:", i)
    done = False
    state = market.reset()

    while not done:
        action = agent.run(state)
        state, wealth, done, j = market.step(action)
        W[i][j] = wealth


plt.plot(market.t_grid, np.mean(W, axis=0).T)
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.title('Average Market Maker Wealth Process')
plt.show()