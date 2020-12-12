import numpy as np
import yaml
import random
import matplotlib
import matplotlib.pyplot as plt
class Agentsinit:
    def __init__(self,AgentsPorperty):
        """
        类初始化，读取配置文件中的参数
        :param AgentsPorperty:
        """
        self.L = AgentsPorperty['Length']
        self.N = AgentsPorperty['Num']
        # self.V = AgentsPorperty['Velocity']
        self.R = AgentsPorperty['R']
        self.step = AgentsPorperty['StepTime']
        self.Noise = AgentsPorperty['Noise']
        self.agentsPos = np.zeros((self.N, 2))
        self.agentsDir = np.zeros((self.N, 2))
        self.agentsVel = np.zeros((self.N, 2))
    #创建初始种群
    def createAggregate(self):
        for i in range(self.N):
            self.agentsPos[i][0] = random.uniform(0, self.L)
            self.agentsPos[i][1] = random.uniform(0, self.L)
            angle = random.uniform(0, 2*np.pi)
            self.agentsDir[i][0] = np.cos(angle)
            self.agentsDir[i][1] = np.sin(angle)
            self.agentsVel[i][0] = random.uniform(-2, 1)
            self.agentsVel[i][1] = random.uniform(-2, 1)
        return self.agentsPos, self.agentsDir, self.agentsVel
    #绘制图像
    def plotAggregate(self, agentsPos, agentsDir):
        agentsPos = np.array(agentsPos, dtype=float)
        agentsDir = np.array(agentsDir, dtype=float)
        plt.figure()
        plt.quiver(agentsPos[:, 0], agentsPos[:, 1], agentsDir[:, 0], agentsDir[:, 1], color= 'b', width=0.005,
                   headwidth=2, scale=30)
        plt.xlim(0,self.L)
        plt.ylim(0,self.L)
        plt.show()





def main():
    config = open('./config.yaml', 'r', encoding='utf-8').read()
    agentconfig = yaml.load(config, Loader=yaml.FullLoader)
    Agents = Agentsinit(agentconfig['alpha_agent'])
    agentsPos, agentsDir,agentsVel = Agents.createAggregate()
    Agents.plotAggregate(agentsPos, agentsDir)
if __name__ == '__main__':
    main()
