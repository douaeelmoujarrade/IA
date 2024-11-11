import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function updates the Q function for a given state-action pair
    following the Q-learning algorithm. It takes as input the Q function, 
    the state-action pair (s, a), the reward r, the next state sprime, 
    the learning rate alpha, and the discount factor gamma.
    It returns the updated Q table.
    """
   
    best_next_action = np.max(Q[sprime])  # Choisir la meilleure action future possible
    Q[s, a] = Q[s, a] + alpha * (r + gamma * best_next_action - Q[s, a])
    return Q

def epsilon_greedy(Q, s, epsilon):
    """
    This function implements the epsilon-greedy algorithm.
    It takes as input the Q table, a state s, and epsilon.
    It returns the action to take following the epsilon-greedy strategy.
    """
    if random.uniform(0, 1) < epsilon:
        # Exploration: choisir une action aléatoire
        return random.randint(0, Q.shape[1] - 1)
    else:
        # Exploitation: choisir l'action avec la valeur Q maximale
        return np.argmax(Q[s])

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="ansi")  

    Q = np.zeros([env.observation_space.n, env.action_space.n])  

    alpha = 0.01 
    gamma = 0.8  
    epsilon = 0.2  

    n_epochs = 20  
    max_itr_per_epoch = 100  
    rewards = []  

    for e in range(n_epochs):
        r = 0  
        S, _ = env.reset() 

        for _ in range(max_itr_per_epoch):
           
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

         
            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma)

          
            S = Sprime

            if done:
                break

        print("Episode #", e, ": Reward =", r)
        rewards.append(r)

    print("Average reward =", np.mean(rewards))

   
    plt.plot(range(n_epochs), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards per Episode')
    plt.show()

    print("Training finished.\n")

   
    env.close()
