import numpy as np
from graph import NodeGraph
import visualize as vsl

def next_states_swap(state):
    result = []
    for i in range(len(state)):
        for j in range(i):
            new_state = np.copy(state)
            new_state[i], new_state[j] = new_state[j], new_state[i]
            result.append(new_state)
    return result

def next_states_reverse(state):
    result = []
    length = len(state)
    for i in range(length):
        for j in range(i - 1):
            new_state = np.copy(state)
            for idx in range((i - j) // 2 + 1):
                new_state[j + idx], new_state[i - idx] = new_state[i - idx], new_state[j + idx]
            result.append(new_state)
    return result

def next_states_insert(state):
    result = []
    length = len(state)
    for i in range(length):
        for j in range(i):
            new_state = np.copy(state)
            for idx in range(j, i):
                new_state[idx], new_state[idx + 1] = new_state[idx + 1], new_state[idx]
            result.append(new_state)
    return result

def climb(state, graph: NodeGraph):
    count = 0
    counts = []
    weights = []
    while True:
        # 选择邻域中最优的状态
        neighbor = min(next_states_swap(state), key=graph.weight)
        if graph.weight(neighbor) >= graph.weight(state):
            vsl.figure(counts, weights, '迭代次数', '总路径权值', '直接登山法优化效果')
            return state
        else:
            # 如果没有最好的状态直接返回
            counts.append(count)
            weights.append(graph.weight(neighbor))
            state = neighbor
            count += 1

def climb_var(state, graph: NodeGraph):
    count = 0
    counts = []
    weights = []
    swap_which = 1
    min_count = 0

    next_states = {
        0: next_states_swap,
        1: next_states_reverse,
        2: next_states_insert,
    }

    while True:
        # 选择邻域中最优的状态
        neighbor = min(next_states[swap_which](state), key=graph.weight)
        if graph.weight(neighbor) >= graph.weight(state):
            if min_count == 2:
                # 所有邻域都是最优, 可以返回
                vsl.figure(counts, weights, '迭代次数', '总路径权值', '多邻域登山法优化效果')
                return state
            else:
                # 如果没有最好的状态, 开始检测是否是所有邻域都最优.更换邻域
                min_count += 1
                swap_which = (swap_which + 1) % 3
        else:
            min_count = 0
            counts.append(count)
            weights.append(graph.weight(neighbor))
            state = neighbor
            count += 1
