from bandit import Bandit
import numpy as np
import torch
bandits = [Bandit() for _ in range(10)]
bandits.append(Bandit(mean=15, variance=4))  # Adding a bandit with different parameters
Q_table = torch.full((len(bandits),), 20.0)  # Initialize Q-values to zero
N_table = torch.zeros(len(bandits))

for i in range(500):
    bandit = bandits[np.argmax(Q_table)]
    reward = bandit.pull()
    N_table[np.argmax(Q_table)] += 1
    Q_table[np.argmax(Q_table)] += (reward - Q_table[np.argmax(Q_table)]) / N_table[np.argmax(Q_table)]
    print(f"Iteration {i+1}: Bandit {np.argmax(Q_table)} pulled, Reward: {reward:.2f}, Q-value: {Q_table[np.argmax(Q_table)]:.2f}")


print("\nFinal Q-values:")
for i, q in enumerate(Q_table):
    print(f"Bandit {i}: Q-value = {q:.2f}, N = {N_table[i]:.0f}")
print("Actual means of bandits:")
for i, bandit in enumerate(bandits):
    print(f"Bandit {i}: Actual Mean = {bandit.mean + bandit.offset:.2f}, Variance = {bandit.variance:.2f}")