from numpy.core.fromnumeric import argmax, size
import scipy
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns


max_car = 20
#动作空间
actions = range(-5, 6)
#折扣
discount = 0.9
#初始值函数


#从状态s到下一个状态的概率
#影响因素：状态s的车辆数，遵从泊松分布的还借车概率
def probability(lamdaRen: int,lamdaRet: int):
    print('状态转移矩阵P 和 期望收益矩阵R 生成中ing')
    #计算车辆的 状态转移矩阵P 和 期望收益矩阵R
    P = np.zeros(shape=[21,21])
    R = np.zeros(shape=[21,21])
    for x in range(21):
        for ren in range(21):
            for ret in range(21):
                Pxx1 = poisson.pmf(ren, lamdaRen) * poisson.pmf(ret, lamdaRet)#从x辆车到x1辆车的转移概率Pxx1 += P借 * P还
                x_after_return = min(20, x+ret)
                x1 = max(x_after_return - ren, 0)
                reward = min(x_after_return, ren) * 10 #收益
                P[x, x1] += Pxx1 
                R[x, x1] += Pxx1 * reward  #收益的期望
        # plt.bar(range(21),R[x])
        # plt.show()            
    #print(P.sum(axis=1))
    return (P, R)

#先借再还
def probability1(lamdaRen: int,lamdaRet: int):
    print('状态转移矩阵P 和 期望收益矩阵R 生成中ing')
    #计算车辆的 状态转移矩阵P 和 期望收益矩阵R
    P = np.zeros(shape=[21,21])
    R = np.zeros(shape=[21,21])
    for x in range(21):
        for ren in range(21):
            for ret in range(21):
                Pxx1 = poisson.pmf(ren, lamdaRen) * poisson.pmf(ret, lamdaRet)#从x辆车到x1辆车的转移概率Pxx1 += P借 * P还
                x_after_rent = max(x-ren,0)
                x1 = min(x_after_rent + ret, 20)
                reward = min(ren, x) * 10 #收益
                P[x, x1] += Pxx1 
                R[x, x1] += Pxx1 * reward  #收益的期望
                
        # plt.bar(range(21),R[x])
        # plt.show()  
    print(R.sum(axis=1))          
    # print(R)
    return (P, R)


Pa, Ra = probability1(3, 3)
Pb, Rb = probability1(4, 2)
if 1:
    # 准备画布大小，并准备多个子图
    _, axes = plt.subplots(2, 2, figsize=(40, 20))
    # 调整子图的间距，wspace=0.1为水平间距，hspace=0.2为垂直间距
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    # 这里将子图形成一个1*6的列表
    axes = axes.flatten()
    fig = sns.heatmap(np.flipud(Pa), cmap="YlGnBu", ax=axes[0])
    # 定义标签与标题
    fig.set_ylabel('#  A State', fontsize=30)
    fig.set_yticks(list(reversed(range(20 + 1))))
    fig.set_xlabel('#  Next A State', fontsize=30)
    fig.set_title('A-Pss', fontsize=30)
    fig = sns.heatmap(np.flipud(Ra), cmap="YlGnBu", ax=axes[1])
    # 定义标签与标题
    fig.set_yticks(list(reversed(range(20 + 1))))
    fig.set_title('A expect Reward', fontsize=30)
    fig = sns.heatmap(np.flipud(Pb), cmap="YlGnBu", ax=axes[2])
    # 定义标签与标题
    fig.set_ylabel('#  B State', fontsize=30)
    fig.set_yticks(list(reversed(range(20 + 1))))
    fig.set_xlabel('#  Next B State', fontsize=30)
    fig.set_title('B-Pss', fontsize=30)
    fig = sns.heatmap(np.flipud(Rb), cmap="YlGnBu", ax=axes[3])
    # 定义标签与标题
    fig.set_yticks(list(reversed(range(20 + 1))))
    fig.set_title('B expect Reward', fontsize=30)
    plt.savefig('./PssRs.png')
 

#一次值迭代
def value_iter(value: np.array,actions: list):
    value1= np.zeros((21,21))
    policy = np.zeros((21,21))
    for A in range(21):
        for B in range(21):
            all_value  = []
            all_action = []
            #计算某一状态下，做了所有动作后的值函数
            for action in actions:
                if A + action <  0 or B - action <  0 or  \
                   A + action > 20 or B - action > 20 : #上下限幅
                    continue
                #移车之后AB两地的车量
                num_car_AB = [A+action, B-action]
                #移车支出
                cost = abs(action) * 2
                #总收益 = A期望收益 + B期望收益
                income = np.sum(Pa[num_car_AB[0]]*Ra[num_car_AB[0]] + Pb[num_car_AB[1]]*Rb[num_car_AB[1]])  
                #状态转移
                Pss1 = np.dot(Pa[num_car_AB[0]].reshape(21,1), Pb[num_car_AB[1]].reshape(1,21))
                #存储所有值函数
                tmpV = income-cost + discount * np.sum(Pss1*value)#某个动作之后的值函数
                all_value.append(tmpV)
                all_action.append(action)
            #值迭代：取最大的值函数作为下一次值函数
            value1[A, B] = max(all_value)
            #更新策略
            policy[A, B] = all_action[argmax(all_value)]
            
    return value1, policy


value = np.zeros((21,21))
policy = np.zeros((21,21))
iterations = 0
# 准备画布大小，并准备多个子图
_, axes = plt.subplots(4, 4, figsize=(40, 20))
# 调整子图的间距，wspace=0.1为水平间距，hspace=0.2为垂直间距
plt.subplots_adjust(wspace=0.1, hspace=0.2)
# 这里将子图形成一个1*6的列表
axes = axes.flatten()
while 1: 
    fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])
    # 定义标签与标题
    fig.set_ylabel('# cars at A location', fontsize=30)
    fig.set_yticks(list(reversed(range(20 + 1))))
    fig.set_xlabel('# cars at B location', fontsize=30)
    fig.set_title('policy {}'.format(iterations), fontsize=30)
    
    last_policy = policy
    last_value = value
    value, policy = value_iter(value, actions)
    print(np.sum(value))  
    iterations += 1
    if  (np.sum(value) - np.sum(last_value) < 1000): #or (last_policy == policy).all():
        print("value_iter_OVER, the policy:")
        break
fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
fig.set_ylabel('# cars at A location', fontsize=30)
fig.set_yticks(list(reversed(range(20 + 1))))
fig.set_xlabel('# cars at B location', fontsize=30)
fig.set_title('optimal value', fontsize=30)
plt.savefig('./figure_4_2.png')
plt.close()