import math
import random
import numpy as np
from graph import NodeGraph
import visualize as vsl

def random_maxmin(graph: NodeGraph):
    result = []
    for i in range(100):
        new_state = np.arange(graph.dimension)
        np.random.shuffle(new_state)
        result.append(graph.weight(new_state))
    return max(result), min(result)

def neighbor_swap(state):
    idx1, idx2 = random.randrange(0, len(state)), random.randrange(0, len(state))
    if idx1 == idx2:
        idx2 = (idx2 + 1) % len(state)
    new_state = np.copy(state)
    new_state[idx1], new_state[idx2] = new_state[idx2], new_state[idx1]
    return new_state

def neighbor_reverse(state):
    a, b = random.randrange(0, len(state)), random.randrange(0, len(state))
    if a == b:
        b = (b + 1) % len(state)
    start = min(a, b)
    end = max(a, b)

    new_state = np.copy(state)

    for idx in range((end - start) // 2 + 1):
        new_state[start + idx], new_state[end - idx] = new_state[end - idx], new_state[start + idx]
    return new_state

def neighbor_insert(state):
    a, b = random.randrange(0, len(state)), random.randrange(0, len(state))
    if a == b:
        b = (b + 1) % len(state)
    src = min(a, b)
    dst = max(a, b)

    new_state = np.copy(state)

    for idx in range(src, dst):
        new_state[idx], new_state[idx + 1] = new_state[idx + 1], new_state[idx]
    return new_state

def simulate_annealing(state, graph: NodeGraph):
    """
    模拟退火算法
    :param state: 初始状态
    :param graph: 图
    :return: 最优状态
    """
    maximum, minimum = random_maxmin(graph)
    temperature = maximum - minimum # 初温取极差
    end_temperature = 1 # 终温
    alpha = 0.98 # 降温系数
    balance_len = 1000 # Markov链长度
    counts = []
    weights = []

    neighbor = {
        0: neighbor_swap,
        1: neighbor_reverse,
        2: neighbor_insert,
    }

    while temperature > end_temperature:

        for i in range(balance_len):
            new_state = neighbor[random.randrange(0, 3)](state) # 随机生成下一个状态(使用随机邻域)
            receive = False
            delta = graph.weight(new_state) - graph.weight(state)
            if delta < 0:
                # 永远接受更好的状态
                receive = True
            else:
                # 概率接受更差的状态
                prob = math.exp(-delta / temperature)
                if random.random() < prob:
                     receive = True

            if receive:
                state = new_state
                counts.append(len(counts) + 1)
                weights.append(graph.weight(state))

        temperature *= alpha
    vsl.figure(counts, weights, '迭代次数', '总路径权值', '模拟退火算法优化效果')
    return state

