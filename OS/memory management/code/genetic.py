import random
import numpy as np
from graph import NodeGraph
import visualize as vsl

class Population:
    def __init__(self, size: int, max_age: int, graph: NodeGraph):
        self.size = size
        self.dimension = graph.dimension
        self.max_age = max_age
        self.graph = graph
        self.mutation_p = 0.05
        self.crossover_p = 0.7

        arr = np.arange(self.dimension)
        self.members = np.vstack([np.random.permutation(arr) for _ in range(self.size)])

    def fitness(self):
        return np.array([1 / self.graph.weight(self.members[idx]) for idx in range(self.size)])

    def crossover(self):
        """
        交叉互换算法
        """
        for i in range(self.size // 5, self.size):
            if random.random() < self.crossover_p:
                j = np.random.randint(0, self.size // 5)
                # 子代取代基因较差的亲代
                self.members[i] = position_based_crossover(self.members[i], self.members[j])

    def select(self):
        """
        基因筛选算法
        """
        selected = np.zeros(self.members.shape, dtype=int)

        # 对种群按照适应度进行降序排序
        indices = self.fitness().argsort()
        self.members = self.members[indices][::-1]

        # 按照顺序分配几何分布概率
        p = 0.7
        prob = np.hstack([(1 - p) ** k * p for k in range(self.size - 1)])
        prob = np.hstack([prob, 1 - prob.sum()])

        # 轮盘赌
        for i in range(self.size):
            choice = np.random.choice(range(self.size), p=prob)
            selected[i] = self.members[choice]

        # 更新种群
        self.members = selected

    def mutate(self):
        """
        基因突变算法
        """
        # 生成随机数列代表是否基因突变
        p_rand = np.random.rand(self.size)
        for i in range(self.size):
            if p_rand[i] < self.mutation_p:
                # 随机抽取两个位置进行区间翻转
                self.members[i] = mutate_reverse(self.members[i])

    def evolve(self):
        best_fitness = 0
        best_one = None
        x_axis, y_axis = [], []
        for age in range(self.max_age):
            self.select()
            self.crossover()
            self.mutate()
            fitness = self.fitness()
            best_idx = fitness.argmax()
            if fitness[best_idx] > best_fitness:
                best_one, best_fitness = self.members[best_idx], fitness[best_idx]
                x_axis.append(age)
                y_axis.append(1 / best_fitness)

            if age % 1000 == 0:
                print(f'age: {age}, weight: {1 / best_fitness}')
        vsl.figure(x_axis, y_axis, '种群代目', '总路径权值', '遗传算法优化效果')
        return best_one


def position_based_crossover(parent_1, parent_2):
    """
    基于位置的交叉互换算法
    :param parent_1: 一个亲本
    :param parent_2: 一个亲本
    :return: 两个亲本交叉互换生成的子代
    """
    length = len(parent_1)
    child = np.zeros(length, dtype=int)
    num = random.randint(1, length)
    pos, val = set(), set()
    work = 0

    # 随机选择区间进行交叉
    for i in range(num):
        while True:
            j = random.randint(0, length - 1)
            if j not in pos:
                break
        child[j] = parent_1[j]
        pos.add(j)
        val.add(parent_1[j])

    # 将剩余的位置按照原有顺序放上去
    for i in range(length):
        if i not in pos:
            while parent_2[work] in val:
                work += 1
            child[i] = parent_2[work]
            work += 1

    return child


def mutate_swap(state):
    idx1, idx2 = random.randrange(0, len(state)), random.randrange(0, len(state))
    if idx1 == idx2:
        idx2 = (idx2 + 1) % len(state)
    new_state = np.copy(state)
    new_state[idx1], new_state[idx2] = new_state[idx2], new_state[idx1]
    return new_state

def mutate_reverse(state):
    a, b = random.randrange(0, len(state)), random.randrange(0, len(state))
    if a == b:
        b = (b + 1) % len(state)
    start = min(a, b)
    end = max(a, b)

    new_state = np.copy(state)

    for idx in range((end - start) // 2 + 1):
        new_state[start + idx], new_state[end - idx] = new_state[end - idx], new_state[start + idx]
    return new_state

def mutate_insert(state):
    a, b = random.randrange(0, len(state)), random.randrange(0, len(state))
    if a == b:
        b = (b + 1) % len(state)
    src = min(a, b)
    dst = max(a, b)

    new_state = np.copy(state)

    for idx in range(src, dst):
        new_state[idx], new_state[idx + 1] = new_state[idx + 1], new_state[idx]
    return new_state
