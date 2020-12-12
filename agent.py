import numpy as np
from matplotlib import pyplot as plt
import copy
from gamma_agent import Gamma_agent
import random
import math
class Agent:
    def __init__(self, num, config, nextPos, nextDir,Vel,gamma_agent):
        """
        :param num: 粒子的编号
        :param config: 配置文件
        :param nextPos: 粒子的下一步坐标
        :param nextDir: 粒子的下一步方向
        :param Vel: 粒子的速度
        :param gamma_agent: 导航粒子
        """
        self.L = config['Length']
        self.N = config['Num']
        self.R = config['R']
        self.T = config['StepTime']
        self.H = config['H']
        self.E = config['Epsilon']
        self.D = config['D']
        self.r = config['K']*self.D
        self.a = config['a']
        self.b = config['b']
        self.c = np.abs(self.a-self.b)/np.sqrt(4*self.a*self.b)
        self.num = num
        self.V = Vel
        self.pos = np.array([0, 0])
        self.dir = np.array([0, 0])
        self.nextPos = nextPos
        self.nextDir = nextDir
        self.gamma_agent = gamma_agent
    #计算两个粒子之间的欧式距离
    def caleuclidDis(self,targetPos):
        euclidDis=np.sqrt((self.nextPos[0] - targetPos[0]) ** 2 + (self.nextPos[1] - targetPos[1]) ** 2)
        return euclidDis
    #计算两个粒子之间的sigam范数
    def calDistance(self,targetPos):
        euclidDis = np.sqrt((self.nextPos[0]-targetPos[0])**2+(self.nextPos[1]-targetPos[1])**2)
        # print(euclidDis)
        sigamDis = (1/self.E)*((math.sqrt(1+self.E*euclidDis**2))-1)
        return sigamDis
    #势能函数导数项的导数项
    def gradDis(self,targetPos):
        euclidDis = math.sqrt((self.nextPos[0] - targetPos[0]) ** 2 + (self.nextPos[1] - targetPos[1]) ** 2)
        # print(euclidDis)
        grad_x =  (targetPos[0]-self.pos[0])/math.sqrt(1+self.E*euclidDis**2)
        grad_y =  (targetPos[1]-self.pos[1])/math.sqrt(1+self.E*euclidDis**2)
        return [grad_x,grad_y]
    #获取邻居节点
    def getNeighbors(self,agentslist):
        Neighborslist=[]
        for i in range(self.N):
            # print(self.calDistance(agentslist[i,:]))
            if self.caleuclidDis(agentslist[i,0:2]) < self.D and i!=self.num:
                if Neighborslist == []:
                    Neighborslist=np.array(agentslist[i,:]).reshape(1, -1)
                else:
                    # print(self.caleuclidDis(agentslist[i,0:2]))
                    Neighborslist = np.vstack((Neighborslist, agentslist[i,:]))
        return Neighborslist
    def roi_h(self,z):
        if z>=0 and z<self.H:
            return 1
        elif z>=self.H and z<=1:
            return 1/2*(1+np.cos(np.pi*(z-self.H)/(1-self.H)))
        else:
            return 0
    #势能函数的导数项
    def grad_term(self,Neighborslist):
        res = np.array([0,0])
        if Neighborslist!=[]:
            for i in range(Neighborslist.shape[0]):
                z = self.calDistance(Neighborslist[i,0:2])
                r_sigam = (1/self.E) * (np.sqrt(1+self.E*self.R**2)-1)
                d_sigam = (1/self.E) * (np.sqrt(1+self.E*self.D**2)-1)
                sigma_z = (z-d_sigam) / np.sqrt(1 + (z - d_sigam) ** 2)
                #由于c=0，所以省去不写
                fei_z = 1 / 2 * ((self.a + self.b) * sigma_z  + (self.a - self.b))
                fei_az = self.roi_h(z / r_sigam) * fei_z
                nij = self.gradDis(Neighborslist[i,0:2])
                temp = [nij[0]*fei_az,nij[1]*fei_az]
                res = res + temp
        else:
            res = np.array([0,0])
        return res
    #速度协同项
    def con_term(self,Neighborslist):
        res = np.array([0,0])
        r_sigam = 1 / self.E * np.sqrt(1 + self.E * self.R ** 2) - 1
        if Neighborslist!=[]:
            for i in range(Neighborslist.shape[0]):
                x = self.roi_h(self.calDistance(Neighborslist[i,0:2])/r_sigam)*(Neighborslist[i,2]-self.V[0])
                y = self.roi_h(self.calDistance(Neighborslist[i,0:2])/r_sigam)*(Neighborslist[i,3]-self.V[1])
                res=res+[x,y]
        else:
            res = np.array([0,0])
        return res
    #导航项
    def nevig_term(self,c1,c2):
        n_x = -c1*(self.pos[0]-self.gamma_agent.pos[0])-c2*(self.V[0]-self.gamma_agent.V[0])
        n_y = -c1*(self.pos[1]-self.gamma_agent.pos[1])-c2*(self.V[1]-self.gamma_agent.V[1])
        return [n_x,n_y]
    #更新粒子状态
    def updateState(self,Neighborslist):
        #深拷贝
        self.pos = copy.deepcopy(self.nextPos)
        self.dir = copy.deepcopy(self.nextDir)
        self.V = self.V + (self.grad_term(Neighborslist)  + self.con_term(Neighborslist) + self.nevig_term(self.gamma_agent.C1,self.gamma_agent.C2))*self.T
        self.nextPos[0] = self.nextPos[0] + self.V[0]*self.T
        self.nextPos[1] = self.nextPos[1] + self.V[1]*self.T
        angle = np.arctan((self.nextPos[0]-self.pos[0])/(self.nextPos[1]-self.pos[1]))
        self.nextDir[0] = np.cos(angle)
        self.nextDir[1] = np.sin(angle)
        return self.nextPos, self.nextDir, self.V
#绘制sigma范数图像
def plotsigamNorm(epsilon):
    sigamDis = []
    z = []
    for i in np.arange(0,10,0.1):
        z.append(i)
        sigamDis.append((1 / epsilon) * ((np.sqrt(1 + epsilon * i ** 2)) - 1))
    plt.figure()
    plt.plot(z,sigamDis)
    plt.show()

def plot_grad_term(e,r,d):
    y = []
    x = []
    res = 0
    r_sigam = (1 / e) * (np.sqrt(1 + e * r ** 2) - 1)
    print(r_sigam)
    for z in np.arange(0,d,0.1):
        sigma_z = (z-d) / np.sqrt(1 + (z-d) ** 2)
        fei_z = 5 * sigma_z
        fei_az = roi_h(z / r_sigam) * fei_z
        res = res + fei_az
        y.append(res)
        x.append(z)
    plt.figure()
    plt.plot(x,y)
    plt.show()
def roi_h(z):
    if z>=0 and z<0.2:
        return 1
    elif z>=0.2 and z<=1:
        return 1/2*(1+np.cos(np.pi*(z-0.2)/(1-0.2)))
    else:
        return 0
if __name__ == "__main__":
    # plotsigamNorm(0.1)
    plot_grad_term(0.1,8.4,25)