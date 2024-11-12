import config
from environment import Environment, Quit, Action
import numpy as np
import random
import pandas
import seaborn
from matplotlib import pyplot


def get_action_eps_greedy_policy(env, q_tab, st, eps):
    prob = random.uniform(0, 1)
    return np.argmax(q_tab[st]) if prob > eps else env.action_space.sample().value


def train(env, num_episodes, max_steps, lr, gamma, eps_min, eps_max, eps_dec_rate): 
    avg_returns = []
    avg_steps = []
    q_tab = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(num_episodes):
        avg_returns.append(0.)
        avg_steps.append(0)
        eps = eps_min + (eps_max - eps_min) * np.exp(-eps_dec_rate * episode) 
        st = env.reset()
        st = env.get_agent_position()[0] * len(env.field_map[0]) + env.get_agent_position()[1]
        for step in range(max_steps):
            act = get_action_eps_greedy_policy(env, q_tab, st, eps)
            new_st, rew, done = env.step(Action(act))
            new_st = new_st[0] * len(env.field_map[0]) + new_st[1]
            q_tab[st][act] = q_tab[st][act] + lr * (rew + gamma * np.max(q_tab[new_st]) - q_tab[st][act])
            if done:
                avg_returns[-1] += rew
                avg_steps[-1] += step + 1
                break
            st = new_st
    return q_tab, avg_returns, avg_steps


def evaluate(num_episodes, max_steps, env, q_tab):
    ep_rew_lst = []
    steps_lst = []
    for episode in range(num_episodes):
        env.reset()
        st = env.get_agent_position()
        st_index = st[0] * len(env.field_map[0]) + st[1]
        step_cnt = 0
        ep_rew = 0
        for step in range(max_steps):
            act = np.argmax(q_tab[st_index])
            new_st, rew, done = env.step(Action(act))
            new_st_index = new_st[0] * len(env.field_map[0]) + new_st[1]
            step_cnt += 1
            ep_rew += rew
            if done:
                break
            st_index = new_st_index 
        ep_rew_lst.append(ep_rew)
        steps_lst.append(step_cnt)

    print(f'TEST Mean reward: {np.mean(ep_rew_lst):.2f}') 
    print(f'TEST STD reward: {np.std(ep_rew_lst):.2f}') 
    print(f'TEST Mean steps: {np.mean(steps_lst):.2f}')


def line_plot(data, name, show):
    pyplot.figure(f'Average {name} per episode: {np.mean(data):.2f}')
    df = pandas.DataFrame({
        name: [np.mean(data[i * 50:(i + 1) * 50]) for i in range(num_episodes // 50)],
        'episode': [50 * i for i in range(num_episodes // 50)]
    })
    plot = seaborn.lineplot(data=df, x='episode', y=name, marker='o', markersize=5, markerfacecolor='red')
    plot.get_figure().savefig(f'{name}.png')
    if show:
        pyplot.show()

num_episodes = 7000
max_steps = 100
learning_rate = 0.05
gamma = 0.95
eps_min = 0.005
eps_max = 1.0
eps_dec_rate = 0.001

environment = Environment('maps/map.txt')
q_table, avg_rewards, avg_steps = train(environment, num_episodes, max_steps, learning_rate, gamma, eps_min, eps_max, eps_dec_rate)

try:
    env = environment
    environment.reset()
    st = environment.get_agent_position()
    environment.render(config.FPS)
    print(q_table)
    
    while True:

        st_index = st[0] * len(environment.field_map[0]) + st[1]
        action = np.argmax(q_table[st_index])
        new_st, _, done = environment.step(Action(action))
        environment.render(config.FPS)
        environment.render_textual()
        print("\n")
        st = new_st
        if done:
            break

    evaluate(num_episodes, max_steps, env, q_table)
    line_plot(avg_rewards, 'Average Rewards', True)
    line_plot(avg_steps, 'Average Steps', True)

except Quit:
    pass