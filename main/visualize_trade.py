# Visualization
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import time

def show_mdd(xs): # xs is cumulative return / portfolio , if reward u should
    # xs = df['reward'].cumsum() / if reward
    i = np.argmax(np.maximum.accumulate(xs) - xs) # end of the period
    j = np.argmax(xs[:i]) # start of period
    plt.figure(figsize=(12,15))
    plt.plot(xs)
    plt.plot([i, j], [xs[i], xs[j]], 'o', color='Red', markersize=10)
    plt.show()

def visualize(info):
    closes = [data[2] for data in info['history']]
    closes_index = [data[1] for data in info['history']]
    # buy tick
    buy_tick = np.array([data[1] for data in info['history'] if data[0] == 0])
    buy_price = np.array([data[2] for data in info['history'] if data[0] == 0])
    sell_tick = np.array([data[1] for data in info['history'] if data[0] == 1])
    sell_price = np.array([data[2] for data in info['history'] if data[0] == 1])

    plt.figure(figsize=(12,15))
    plt.plot(closes_index, closes)
    plt.scatter(buy_tick, buy_price - 2, c='g', marker="^", s=20)
    plt.scatter(sell_tick, sell_price + 2, c='r', marker="v", s=20)
    plt.show(block=True)
    time.sleep(3)

def get_file(dir):
  list_of_files = glob.glob(dir+'*') # * means all if need specific format then *.csv
  latest_file = max(list_of_files, key=os.path.getctime)
  return str(latest_file)

FILENAME = get_file('./deep_rl_trader/main/info/')
info = np.load(FILENAME, allow_pickle=True).all()
visualize(info)