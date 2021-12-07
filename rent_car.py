import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import poisson

# 每个场的车容量
MAX_CARS = 20

# A场租车请求的平均值
RENTAL_REQUEST_FIRST_LOC = 3
# B场租车请求的平均值
RENTAL_REQUEST_SECOND_LOC = 4

# A场还车请求的平均值
RETURNS_FIRST_LOC = 3
# B场还车请求的平均值
RETURNS_SECOND_LOC = 2

# 收益折扣
DISCOUNT = 0.9
# 租车收益
RENTAL_CREDIT = 10
# 移车支出
MOVE_CAR_COST = 2

# （移动车辆）动作空间：【-5，5】
actions = np.arange(-5, 5 + 1)

# 租车还车的数量满足一个poisson分布，限制由泊松分布产生的请求数大于POISSON_UPPER_BOUND时其概率压缩至0
POISSON_UPPER_BOUND = 20
# 存储每个（n,lamda）对应的泊松概率
poisson_cache = dict()

def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam # 定义唯一key值，除了索引没有实际价值
    if key not in poisson_cache:
        # 计算泊松概率，这里输入为n与lambda，输出泊松分布的概率质量函数，并保存到poisson_cache中
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]

#
#参数1：state   AB两地的车辆数
#参数2：action  一个动作
#参数3：value   返回的值函数
def update_value(state, action, value):

    #晚上挪玩车后，AB两地车的数量
    num_car_AB = [0, 0]
    
    #值函数
    V = 0.0
    
    # 移动车辆产生负收益
    V -= abs(action) * 2

    # 移动后的车辆总数不能超过20
    num_car_AB[0] = min(state[0] - action, 20)
    num_car_AB[1] = min(state[1] + action, 20)

    # 遍历两地全部的可能概率下（<11）租车请求数目
    for i in range(11):
        for j in range(11):
            # prob为两地租车请求的联合概率
            prob = poisson_probability(i, 3) * poisson_probability(j, 4)

            # 两地原本的车的数量
            num_car_A = num_car_AB[0]
            num_car_B = num_car_AB[1]

            # 有效的租车数目必须小于等于该地原有的车辆数目
            Arents = min(num_car_A, i)
            Brents = min(num_car_B, j)
            
            # 更新借车后的车辆数
            num_car_A -= Arents
            num_car_B -= Brents
            
            # 计算回报, 租车数目 * 10
            reward = (Arents + Brents) * 10      

            # 两地的还车数目均为泊松分布均值
            # 更新还车后的车辆数
            num_car_A = min(num_car_A + 3, 20)
            num_car_B = min(num_car_B + 2, 20)
            
            # 策略评估：V(s) += p(a...|s) * [r + γV(s')]
            V += prob * (reward + DISCOUNT * value[num_car_A, num_car_B])
    return V


def policy_iter():
    value = np.zeros((21, 21))
    policy = np.zeros(value.shape, dtype=np.int)
   
    # 设置迭代参数
    iterations = 0
    
    # 准备画布大小，并准备多个子图
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    
    
    # 更新策略
    while True:
        fig = sns.heatmap(np.flipud(policy), cmap="rainbow", ax=axes[iterations], annot=True)
               
        # 定义标签与标题
        fig.set_ylabel('# cars at A', fontsize=30)
        fig.set_yticks(list(reversed(range(21))))
        fig.set_xlabel('# cars at B', fontsize=30)
        fig.set_title(f'policy {iterations}', fontsize=30)

        # 收敛值函数
        while True:
            old_value = value.copy()
            #一轮值迭代,得到新的value
            for i in range(21):
                for j in range(21):
                    value[i, j] = update_value([i, j], policy[i, j], value)
                    
            # 收敛后退出循环
            max_value_change = abs(old_value - value).max()
            print(f'max value change: {max_value_change}')
            if max_value_change < 1e-4:
                break
        
        policy_stable = True
        # i、j分别为两地现有车辆总数
        for i in range(21):
            for j in range(21):
                old_action = policy[i, j]
                action_returns = []
                #对每一个动作
                for action in actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(update_value([i, j], action, value))
                    else:
                        action_returns.append(-np.inf)
                #贪婪策略，找出产生最大动作价值的动作
                new_action = actions[np.argmax(action_returns)]
                # 更新策略
                policy[i, j] = new_action
                if policy_stable and old_action != new_action:#策略发生了更新
                    policy_stable = False
                    
        print(f'policy stable {policy_stable}')

        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap="rainbow_r", ax=axes[-1], annot=False)
            fig.set_ylabel('# cars at A', fontsize=30)
            fig.set_yticks(list(reversed(range(21))))
            fig.set_xlabel('# cars at B', fontsize=30)
            fig.set_title('optimal value', fontsize=30)
            break

        iterations += 1

    plt.savefig('./policy_iter.png')


def value_iter():
    # value matrix start from 0
    value = np.zeros((21, 21))
    # policy matrix start from 0
    policy = np.zeros(value.shape, dtype=np.int16)
    # iteration times
    iteration = 0
    
    _, axes = plt.subplots(5, 4, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    axes = axes.flatten()
    while True:        
        # draw currunt policy 
        fig = sns.heatmap(np.flipud(policy), annot=True, linewidths= 0.2, cmap="rainbow", ax=axes[iteration]) 
        fig.set_ylabel('# cars at A', fontsize=20)
        fig.set_yticks(list(reversed(range(21))))
        fig.set_xlabel('# cars at B', fontsize=20)
        fig.set_title('policy {}'.format(iteration), fontsize=20)
        
        old_value = value.copy()
        # Value Iteration
        policy_stable = True  #judge stop improvement
        for i in range(21):
            for j in range(21):
                action_returns = []
                for action in actions:
                    if (0 <= action <= i) or (-j <= action <= 0):
                        action_returns.append(update_value([i, j], action, value))
                    else:
                        action_returns.append(-np.inf)
                # 贪婪策略，选取最大的
                new_action = actions[np.argmax(action_returns)]
                # upload policy
                policy[i, j] = new_action
                new_state_value = update_value([i, j], policy[i, j], value)
                value[i, j] = new_state_value
        
        max_value_change = abs(old_value - value).max()
        print(f'max value change {max_value_change}')
        if max_value_change > 4 :
            policy_stable = False
        print(f'policy stable {policy_stable}')

        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap="rainbow_r", ax=axes[-1])
            fig.set_ylabel('# cars at B', fontsize=20)
            fig.set_yticks(list(reversed(range(21))))
            fig.set_xlabel('# cars at B', fontsize=20)
            fig.set_title('optimal value', fontsize=20)
            break

        iteration += 1

    plt.savefig('./value_iter.png')
    plt.close()
if __name__ == '__main__':
    value_iter()
    policy_iter()