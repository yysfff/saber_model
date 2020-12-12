import numpy as np

class Gamma_agent:
    def __init__(self, config, position):
        self.pos = config["Position"]
        self.V = config["Velocity"]
        self.T = config["StepTime"]
        self.C1 = config["C1"]
        self.C2 = config["C2"]
    #更新导航粒子的坐标
    def updateState(self):
        self.pos[0] = self.pos[0] + self.V[0] * self.T
        self.pos[1] = self.pos[1] + self.V[1] * self.T