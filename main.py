import numpy as np
import yaml
from matplotlib import pyplot as plt
from agent import Agent
from agentinit import Agentsinit
from gamma_agent import Gamma_agent
def main():
    config = open('./config.yaml', 'r', encoding='utf-8').read()#读取配置文件
    agentconfig = yaml.load(config, Loader=yaml.FullLoader)#加载到agentconfig
    alpha_agentconfig=agentconfig['alpha_agent']#加载alpha_agent的参数
    gamma_agentconfig=agentconfig['gamma_agent']#加载gamma_agent的参数
    Agents = Agentsinit(alpha_agentconfig)#初始化alpha_agent
    agentsPos, agentsDir, agentsVel= Agents.createAggregate()#创建初始alpha集群
    gamma_agent=Gamma_agent(gamma_agentconfig, gamma_agentconfig['Position'])#创建初始gamma_agent
    Pos = np.array([0, 0])#初始化alpha_agent的坐标
    Dir = np.array([0, 0])#初始化alpha_agent的方向
    Vel = np.array([0, 0])#初始化alpha_agent的速度
    agentlist = np.hstack((agentsPos,agentsVel))#合并所有粒子的坐标和速度信息，作为判断邻居节点的输入参数
    #运行10000步
    for step in range(10001):
        for i in range(alpha_agentconfig['Num']):
            #更新粒子状态
            agent = Agent(i,alpha_agentconfig,agentsPos[i,:],agentsDir[i,:],agentsVel[i,:], gamma_agent)
            #获取邻居节点
            Neighborslist = agent.getNeighbors(agentlist)
            #根据邻居节点更新自己的状态
            pos, dir, vel= agent.updateState(Neighborslist)
            #将每个粒子的信息存进数组
            Pos = np.vstack((Pos, pos))
            Dir = np.vstack((Dir, dir))
            Vel = np.vstack((Vel, vel))
        #删除首行空信息
        Pos = np.delete(Pos, 0, axis=0)
        Dir = np.delete(Dir, 0, axis=0)
        Vel = np.delete(Vel, 0, axis=0)
        #更新gamma_agent状态
        gamma_agent.updateState()
        #每100步生成一副图像
        if (step % 100 == 0):
            plt.figure()
            plt.scatter(Pos[:, 0], Pos[:, 1],color='b');
            plt.scatter(gamma_agent.pos[0],gamma_agent.pos[1],ec='g',fc='g')
            #以下为对距离保持一定范围内的两个粒子之间绘制直线，这里选取距离范围为（alpha_agentconfig['D']-1,alpha_agentconfig['D']+1）
            for i in range(alpha_agentconfig['Num']):
                for j in range(alpha_agentconfig['Num']):
                    if calDistance(Pos[i,:],Pos[j,:],0.1)>=fanshu(0.1,alpha_agentconfig['D'])-1 and \
                            calDistance(Pos[i,:],Pos[j,:],0.1)<=fanshu(0.1,alpha_agentconfig['D'])+1:
                        dx = Pos[j,0]-Pos[i,0]
                        dy = Pos[j,1]-Pos[i,1]
                        plt.arrow(Pos[i,0],Pos[i,1],dx,dy,ec='r',fc='r')
            plt.title(f'step={step},r={alpha_agentconfig["R"]}')
            plt.show()
        #更新alpha—agent初始化参数
        agentsPos = Pos
        agentsDir = Dir
        agentsVel = Vel
        #更新agentlist
        agentlist = np.hstack((agentsPos, agentsVel))
        #清空上一步粒子信息缓存
        Pos = np.array([0, 0])
        Dir = np.array([0, 0])
        Vel = np.array([0, 0])
def calDistance(selfPos, targetPos,epsilon):
    euclidDis = np.sqrt((selfPos[0]-targetPos[0])**2+(selfPos[1]-targetPos[1])**2)
    sigamDis = (1/epsilon)*((np.sqrt(1+epsilon*euclidDis**2))-1)
    return sigamDis
def fanshu(epsilon,x):
    return (1/epsilon)*((np.sqrt(1+epsilon*x**2))-1)
if __name__ == '__main__':
    main()
