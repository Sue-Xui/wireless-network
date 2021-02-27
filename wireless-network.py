import random
import time
import datetime

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np


class myfirst(object):

    def MCF(self):
        totalcost = 0  # 总成本
        contribution = 0  # 用于计算参与者平均贡献值
        totalDis = 0  # 用于计算总移动距离
        y = sorted(self.uavlis, key=lambda x: x["f"])
        # for i in range(self.uavs):
        #     print(y[i])
        temptask = [0 for i in range(self.tasks)]
        t = 0
        u = 0
        while t < self.tasks:
            for i in range(self.uavs):
                tempabi = y[i]["abi"]  # 剩余执行能力
                tempdis = y[i]["distance"]  # 剩余飞行距离
                currentl = 0  # 当前位置
                for j in range(1, self.tasks + 1):
                    if tempabi >= self.tasklis[j]["q"] and temptask[j - 1] == 0 and tempdis >= self.tasklis[currentl][
                        j]:
                        temptask[j - 1] = 1
                        t += 1
                        totalDis += self.tasklis[currentl][j]
                        contribution += self.tasklis[j]["q"]
                        tempabi -= self.tasklis[j]["q"]
                        tempdis -= self.tasklis[currentl][j]
                        totalcost = totalcost + y[i]["f"] + y[i]["p"] * self.tasklis[currentl][j]  # 一次任务总成本
                        currentl = j  # 变换当前位置
                if tempabi != y[i]["abi"]:
                    u += 1
            break

        self.MCFCost = round(totalcost)
        self.MCFAverT = round(totalcost / (t))
        self.MCFAverCon = round(contribution / u)
        self.MCFDis = round(totalDis)
        # print("MCF总成本%d"%(self.MCFCost))

    def MRF(self):
        totalcost = 0  # 总成本
        contribution = 0  # 用于计算参与者平均贡献值
        totalDis = 0
        z = sorted(self.uavlis, key=lambda x: x["ratio"], reverse=True)
        # for i in range(self.uavs):
        #     print(z[i])
        temptask = [0 for i in range(self.tasks)]
        t = 0
        u = 0
        while t < self.tasks:
            for i in range(self.uavs):
                tempabi = z[i]["abi"]  # 剩余执行能力
                tempdis = z[i]["distance"]  # 剩余飞行距离
                currentl = 0  # 当前位置
                for j in range(1, self.tasks + 1):
                    if tempabi >= self.tasklis[j]["q"] and temptask[j - 1] == 0 and tempdis >= self.tasklis[currentl][
                        j]:
                        temptask[j - 1] = 1
                        t += 1
                        totalDis += self.tasklis[currentl][j]
                        contribution += self.tasklis[j]["q"]
                        tempabi -= self.tasklis[j]["q"]
                        tempdis -= self.tasklis[currentl][j]
                        totalcost = totalcost + z[i]["f"] + z[i]["p"] * self.tasklis[currentl][j]  # 一次任务总成本
                        currentl = j  # 变换当前位置
                if tempabi != z[i]["abi"]:
                    u += 1
            break

        self.MRFCost = round(totalcost)
        self.MRFAverT = round(totalcost / t)
        self.MRFAverCon = round(contribution / u)
        self.MRFDis = round(totalDis)
        # print("MRF总成本%d"%(self.MRFCost))


    def GA(self):
        # 开始迭代
        lines = [[0 for i in range(self.tasks)] for i in range(self.size)]

        # 适应度
        fit = [0 for i in range(self.size)]

        # 初始种群，计算适应度

        for i in range(0, self.size):
            j = 0
            while j < self.tasks:
                num = int(random.uniform(0, self.tasks)) + 1
                if self.isHas(lines[i], num) == False:
                    lines[i][j] = num
                    j += 1
            # 计算适应度
            fit[i] = self.calfitness(lines[i])

        # 迭代次数
        t = 0

        while t < self.times:
            # 适应度
            newlines = [[0 for i in range(self.tasks)] for i in range(self.size)]
            nextfit = [0 for i in range(self.size)]
            randomfit = [0 for i in range(self.size)]
            totalfit = 0
            tmpfit = 0

            for i in range(self.size):
                totalfit += fit[i]

            # 通过适应度占总适应度的比例生成随机适应度
            for i in range(self.size):
                randomfit[i] = tmpfit + fit[i] / totalfit
                tmpfit += randomfit[i]

            # 上一代中的最优直接遗传到下一代
            m = fit[0]
            ml = 0

            for i in range(self.size):
                if m < fit[i]:
                    m = fit[i]
                    ml = i

            for i in range(self.tasks):
                newlines[0][i] = lines[ml][i]

            nextfit[0] = fit[ml]

            # 对最优解使用爬山算法促使其自我进化
            self.mountain(newlines[0])

            # 开始遗传
            nl = 1
            while nl < self.size/2:
                # 根据概率选取排列
                r = int(self.randomSelect(randomfit))

                # 判断是否需要交叉，不能越界
                if random.random() < self.JCL and nl + 1 < self.size:
                    fline = [0 for x in range(self.tasks)]
                    nline = [0 for x in range(self.tasks)]

                    # 获取交叉排列
                    rn = int(self.randomSelect(randomfit))

                    f = int(random.uniform(0, self.tasks))
                    l = int(random.uniform(0, self.tasks))

                    min = 0
                    max = 0
                    fpo = 0
                    npo = 0

                    if f < l:
                        min = f
                        max = l
                    else:
                        min = l
                        max = f

                    # 将截取的段加入新生成的基因
                    while min < max:
                        fline[fpo] = lines[rn][min]
                        nline[npo] = lines[r][min]

                        min += 1
                        fpo += 1
                        npo += 1

                    for i in range(self.tasks):
                        if self.isHas(fline, lines[r][i]) == False:
                            fline[fpo] = lines[r][i]
                            fpo += 1

                        if self.isHas(nline, lines[rn][i]) == False:
                            nline[npo] = lines[rn][i]
                            npo += 1

                    # 基因变异
                    self.change(fline)
                    self.change(nline)

                    # 交叉并且变异后的结果加入下一代
                    for i in range(self.tasks):
                        newlines[nl][i] = fline[i]
                        newlines[nl + 1][i] = nline[i]

                    nextfit[nl] = self.calfitness(fline)
                    nextfit[nl + 1] = self.calfitness(nline)

                    nl += 2
                else:
                    # 不需要交叉的，直接变异，然后遗传到下一代
                    line = [0 for i in range(self.tasks)]
                    i = 0
                    while i < self.tasks:
                        line[i] = lines[r][i]
                        i += 1

                    # 基因变异
                    self.change(line)

                    # 加入下一代
                    i = 0
                    while i < self.tasks:
                        newlines[nl][i] = line[i]
                        i += 1

                    nextfit[nl] = self.calfitness(line)
                    nl += 1

            # 新的一代覆盖上一代
            for i in range(self.size):
                for h in range(self.tasks):
                    lines[i][h] = newlines[i][h]

                fit[i] = nextfit[i]

            t += 1

        # 上代中最优的为适应函数最小的
        m = fit[0]
        ml = 0

        for i in range(self.size):
            if m < fit[i]:
                m = fit[i]
                ml = i

        # 迭代完成
        # 输出结果:
        self.calfitness(lines[ml], True, False)

    # 是否包含路线
    def isHas(self, line, num):
        for i in range(self.tasks):
            if line[i] == num:
                return True
        return False

    def mountain(self, line):
        oldFit = self.calfitness(line, False,False)

        i = 0
        while i < self.PSCS:
            f = random.uniform(0, self.tasks)
            n = random.uniform(0, self.tasks)

            tmp = line[int(f)]
            line[int(f)] = line[int(n)]
            line[int(n)] = tmp

            newFit = self.calfitness(line, False,False)

            if newFit < oldFit:
                tmp = line[int(f)]
                line[int(f)] = line[int(n)]
                line[int(n)] = tmp
            i += 1

    # 基因变异
    def change(self, line):
        if random.random() < self.BYL:
            i = 0
            while i < self.JYHW:
                f = random.uniform(0, self.tasks)
                n = random.uniform(0, self.tasks)

                tmp = line[int(f)]
                line[int(f)] = line[int(n)]
                line[int(n)] = tmp

                i += 1

    # 根据概率随机选择的序列
    def randomSelect(self, ranFit):
        ran = random.random()
        for i in range(int(self.size/2)):
            if ran < ranFit[i]:
                return i

    def __init__(self, size, times, tasks, uavs, pw):
        self.JCL = 0.9  # 遗传交叉率
        self.BYL = 0.1  # 遗传变异率
        self.JYHW = 5  # 遗传换位次数
        self.PSCS = 10  # 爬山算法迭代次数
        self.size = size  # 规模
        self.times = times  # 迭代次数
        self.tasks = tasks  # 任务数量
        self.uavs = uavs  # 用户总数
        self.pw = pw  # 惩罚因子
        self.uavlis = []
        # MCF算法
        self.MCFAverT = 0
        self.MCFAverCon = 0
        self.MCFCost = 0
        self.MCFDis = 0
   #     self.MCFtime = 0
        # MRF算法
        self.MRFCost = 0
        self.MRFAverT = 0
        self.MRFAverCon = 0
        self.MRFDis = 0
 #       self.MRFDis = 0
        # GA算法
        self.GACost = 0
        self.GAAverT = 0
        self.GAAverCon = 0
        self.GADis = 0
    #    self.GAtime = 0
        #GA2
        self.GACost2 = 0
        self.GAAverT2 = 0
        self.GAAverCon2 = 0
        self.GADis2 = 0
        #SA
        self.SACost = 0
        self.SAAverT = 0
        self.SAAverCon = 0
        self.SADis = 0

        self.run()

    def run(self):
        self.init()
        self.MCF()
        self.MRF()
        self.SA()
        self.GA()
        self.GA2()

def draw(y_data1, y_data2, y_data3,y_data4,y_data5, flag):
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 18,
            }

    plt.tick_params(labelsize=13)
    x_data = np.linspace(2, 40, 20, endpoint=True)
    plt.xlabel("The number of UAVs", font)
    if flag == 1:
        plt.ylabel("Total sensing cost", font)
    elif flag == 2:
        plt.ylabel("Average cost per task", font)
    elif flag == 3:
        plt.ylabel("Average contribution ", font)
    elif flag == 4:
        plt.ylabel("Total moving distance ", font)
    else:
        plt.ylabel("execution time(ms)", font)
    plt.plot(x_data, y_data1, marker='o', label='MCF')
    plt.plot(x_data, y_data2, marker='*', label='MRF')
    plt.plot(x_data, y_data3, marker='x', label='GA-TA')
    #plt.plot(x_data, y_data4, marker='v', label='GA')
    plt.plot(x_data, y_data5, marker='.', label='SA')
    plt.legend(fontsize='large')
    plt.grid(linestyle='-.')
    plt.grid(True)  # 生成网格

    plt.savefig('./images/' + str(flag) + '.pdf', dpi=300)
    plt.show()

    plt.close(0)

def drawga(y_data3,y_data4):
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 18,
            }

    plt.tick_params(labelsize=13)
    plt.ylim(400,1400)
    x_data = np.linspace(2, 40, 20, endpoint=True)
    plt.xlabel("The number of UAVs", font)
    plt.ylabel("Total sensing cost", font)
    plt.plot(x_data, y_data3, marker='x', label='GA-TA')
    plt.plot(x_data, y_data4, marker='v', label='GA')
    plt.legend(fontsize='large')
    plt.grid(linestyle='-.')
    plt.grid(True)  # 生成网格

    plt.savefig('./images/' + '5.pdf', dpi=300)
    plt.show()

    plt.close(0)

if __name__ == "__main__":
    y_data1 = []
    y_data2 = []
    y_data3 = []
    y_data4 = []
    y_data5 = []
    y_data6 = []
    y_data7 = []
    y_data8 = []
    y_data9 = []
    y_data10 = []
    y_data11 = []
    y_data12 = []
    y_data13 = []
    y_data14 = []
    y_data15 = []
    y_data16 = []
    y_data17 = []
    y_data18 = []
    y_data19 = []
    y_data20 = []
    for i in range(2, 42, 2):
        print("当用户数为%d时" % i)
        test = myfirst(100, 150, 10, i, 1000)
        y_data1.append(test.MCFCost)
        y_data2.append(test.MRFCost)
        y_data3.append(test.GACost)
        y_data13.append(test.GACost2)
        y_data17.append(test.SACost)
        y_data4.append(test.MCFAverT)
        y_data5.append(test.MRFAverT)
        y_data6.append(test.GAAverT)
        y_data14.append(test.GAAverT2)
        y_data18.append(test.SAAverT)
        y_data7.append(test.MCFAverCon)
        y_data8.append(test.MRFAverCon)
        y_data9.append(test.GAAverCon)
        y_data15.append(test.GAAverCon2)
        y_data19.append(test.SAAverCon)
        y_data10.append(test.MCFDis)
        y_data11.append(test.MRFDis)
        y_data12.append(test.GADis)
        y_data16.append(test.GADis2)
        y_data20.append(test.SADis)

    draw(y_data1, y_data2, y_data3,y_data13,y_data17, 1)
    draw(y_data4, y_data5, y_data6,y_data14,y_data18, 2)
    draw(y_data7, y_data8, y_data9,y_data15,y_data19, 3)
    draw(y_data10, y_data11, y_data12,y_data16,y_data20, 4)
    drawga(y_data3,y_data13)

