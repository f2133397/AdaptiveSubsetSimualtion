import time

import matplotlib.pyplot as plt
import math
import seaborn as sns
import geatpy as ea
from scipy.special import ndtri
import scipy.stats as st
import warnings
import pandas as pd
import numpy as np
import itertools
import sys
import os

sys.path.append("\\".join(sys.path[0].split('\\')[:-1]))
from itertools import combinations
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import lgsvl
from environs import Env
import itertools

from scipy.special import ndtri
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from func_timeout import func_set_timeout
import func_timeout


@func_set_timeout(10)
def run_simulation(sim: lgsvl.Simulator, ego_uid):
    # sim.run(time_limit=5,time_scale = 2)
    sim.run(time_limit=0.1, time_scale=2)
    # 得到包括所有agent的列表
    all_agents = sim.get_agents()
    for agent in all_agents:
        if agent.uid == ego_uid:
            ego = agent
    ttc = 9999
    # 计算ego和其它车辆的TTC
    for agent in all_agents:
        if agent.uid != ego_uid:
            # 判断车辆是否在ego的前方
            if ego.state.transform.position.x > agent.state.transform.position.x:  # ego在后方  agent在前方
                if ego.state.velocity.x < agent.state.velocity.x:  # ego的速度比agent大
                    # 计算TTC
                    ttc_temp = (- agent.state.transform.position.x + ego.state.transform.position.x-5) / (
                                agent.state.velocity.x - ego.state.velocity.x)
                    if ttc_temp < ttc and ttc >=0:
                        ttc = ttc_temp

    if ttc <0:
        print('debug')
    return ttc


class test_apolo:
    def __init__(self, env: Env, sim: lgsvl.Simulator, BRIDGE_HOST, BRIDGE_PORT):
        self.env = env
        self.sim = sim
        self.BRIDGE_HOST = BRIDGE_HOST
        self.BRIDGE_PORT = BRIDGE_PORT
        self.sim_num = 0
        self.car_npc = {}
        self.base_point = lgsvl.Vector(1500, 0, -17.9)

    def test(self, scenarios=None):
        # 通过sim.get_agents()获取到当前场景中的所有agent
        agents = self.sim.get_agents()
        # 如果agents为空，说明是第一次运行，需要初始化ego
        if len(agents) == 0:
            state = lgsvl.AgentState()
            state.transform.position = self.base_point
            state.transform.rotation = lgsvl.Vector(0, 270, 0)
            self.forward = lgsvl.utils.transform_to_forward(state)  # 获取到前向的方向角
            state.velocity = 10 * self.forward

            self.ego = self.sim.add_agent(
                self.env.str("LGSVL__VEHICLE_0", "617ff042-1c37-4ff5-af08-598fd1057a26"),
                lgsvl.AgentType.EGO, state)
            self.ego.connect_bridge(self.BRIDGE_HOST, self.BRIDGE_PORT)  # 加入本车并且完成桥接
            self.ego_uid = self.ego.uid  # 保存本车的uid
            # Dreamview setup
            dv = lgsvl.dreamview.Connection(self.sim, self.ego, self.BRIDGE_HOST)

            dv.set_destination(-1500, -17.9, 0)

            modules = [
                'Localization',
                # 'Perception',
                'Transform',
                'Routing',
                'Prediction',
                'Planning',
                # 'Camera',
                'Traffic Light',
                'Control'
            ]
            # dv.check_module_status(modules)
            for module in modules:
                dv.enable_module(module)

        # 遍历scenarios的每一行
        result = np.array([])
        for scenario in scenarios:
            # 判断car_npc是否为空
            # if not self.car_npc == {}:
            #     self.sim.remove_agent(self.car_npc['npc2'])
            #     del self.car_npc['npc2']
            #     self.sim.remove_agent(self.car_npc['npc3'])
            #     del self.car_npc['npc3']
            try:
                self.sim.remove_agent(self.car_npc['npc2'])
                del self.car_npc['npc2']
                self.sim.remove_agent(self.car_npc['npc3'])
                del self.car_npc['npc3']
                #print("已删除npc2和npc3")
            except:
                print("npc2或者npc3不存在")

            # 将scenario中小于0的值变为0
            # for i in range(len(scenario)):
            #     if scenario[i] < 0:
            #         scenario[i] = 0
            # result增加一个0
            result = np.append(result, 0)
            # v3 = scenario[2]
            # v2 = scenario[1]
            # v1 = scenario[0]
            # x2 = self.base_point.x - scenario[3] - 5
            # y2 = self.base_point.z + scenario[5]  # 根据车身长度偏移
            # x3 = x2 - scenario[4] - 5
            # y3 = self.base_point.z

            # v3 = scenario[2]当且仅当scenario[2]>=0时，否则v3=0
            v3 = scenario[2] if scenario[2] >= 0 else 0
            # 下面都是如此
            v2 = scenario[1] if scenario[1] >= 0 else 0
            v1 = scenario[0] if scenario[0] >= 0 else 0
            x2 = self.base_point.x - scenario[3] - 5 if scenario[3] >= 0 else self.base_point.x - 5
            y2 = self.base_point.z + scenario[5] if scenario[5] >= 0 else self.base_point.z
            x3 = x2 - scenario[4] - 5 if scenario[4] >= 0 else x2 - 5
            y3 = self.base_point.z

            print("当前场景参数：" + str(scenario))
            # parameters = np.append(parameters, scenario[0])
            s = lgsvl.AgentState()
            s.velocity = self.forward * v1
            s.transform.position = self.base_point
            s.transform.rotation = lgsvl.Vector(0, 270, 0)
            self.ego.state = s
            state = lgsvl.AgentState()
            state.transform.position = lgsvl.Vector(x2, 0, y2)
            state.transform.rotation = lgsvl.Vector(0, 270, 0)

            npc2 = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)
            waypoints = []
            loc = lgsvl.Vector(x2 - 50, 0, y2 - 1.75)
            wp = lgsvl.DriveWaypoint(
                position=loc, speed=v2, angle=lgsvl.Vector(0, 270, 0)
            )

            wp1 = lgsvl.DriveWaypoint(
                position=lgsvl.Vector(x2 - 100, 0, y2 - 3.5), speed=v2, angle=lgsvl.Vector(0, 270, 0)
            )

            wp2 = lgsvl.DriveWaypoint(
                position=lgsvl.Vector(x2 - 500, 0, y2 - 3.5), speed=v2, angle=lgsvl.Vector(0, 270, 0)
            )

            waypoints.append(wp)
            waypoints.append(wp1)
            waypoints.append(wp2)
            npc2.follow(waypoints)
            self.car_npc['npc2'] = npc2

            state = lgsvl.AgentState()
            state.transform.position = lgsvl.Vector(x3, 0, y3)
            state.transform.rotation = lgsvl.Vector(0, 270, 0)
            state.velocity = self.forward * v3
            npc3 = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)

            self.car_npc['npc3'] = npc3

            vehicles = {
                self.ego: "EGO",
                npc2: "Sedan",
                npc3: "Sedan",
            }

            def on_collision(agent1, agent2, contact):
                name1 = vehicles[agent1]
                name2 = 'Sedan' if agent2 is not None else "OBSTACLE"
                if name2 == 'Sedan':
                    # print("{} collided with {} at {}".format(name1, name2, contact))
                    result[-1] = 1
                    # result = 1

            self.ego.on_collision(on_collision)
            # self.sim.run(time_limit=5)
            try:
                # run_simulation(self.sim)
                TTC = 999999
                for i in range(0, 50):
                    ttc_temp = run_simulation(self.sim, self.ego_uid)
                    if ttc_temp < TTC:
                        TTC = ttc_temp
                # 如果没有发生碰撞，result里替换为TTC
                if result[-1] == 0:
                    result[-1] = - TTC

                print(result[-1])

            except func_timeout.exceptions.FunctionTimedOut as e:
                print(e)
                print("Time out!!!")
                print("请重新开启svl")
                name = input("请输入")
                self.sim = lgsvl.Simulator("127.0.0.1", 8181)
                if self.sim.current_scene_id == "085d664e-f3f4-4f38-b303-d45c25125297":
                    self.sim.reset()  # 重置仿真
                else:
                    # sim.load("BorregasAve")
                    self.sim.load("085d664e-f3f4-4f38-b303-d45c25125297")

                state = lgsvl.AgentState()
                state.transform.position = self.base_point
                state.transform.rotation = lgsvl.Vector(0, 270, 0)
                self.forward = lgsvl.utils.transform_to_forward(state)  # 获取到前向的方向角
                state.velocity = 10 * self.forward

                self.ego = self.sim.add_agent(
                    self.env.str("LGSVL__VEHICLE_0", "617ff042-1c37-4ff5-af08-598fd1057a26"),
                    lgsvl.AgentType.EGO, state)
                self.ego.connect_bridge(self.BRIDGE_HOST, self.BRIDGE_PORT)  # 加入本车并且完成桥接
                # Dreamview setup
                dv = lgsvl.dreamview.Connection(self.sim, self.ego, self.BRIDGE_HOST)
                self.ego_uid = self.ego.uid  # 保存本车的uid
                dv.set_destination(-1500, -17.9, 0)

                modules = [
                    'Localization',
                    # 'Perception',
                    'Transform',
                    'Routing',
                    'Prediction',
                    'Planning',
                    # 'Camera',
                    'Traffic Light',
                    'Control'
                ]
                # dv.check_module_status(modules)
                for module in modules:
                    dv.enable_module(module)

                print("当前场景参数：" + str(scenario))
                # parameters = np.append(parameters, scenario[0])
                s = self.ego.state
                s.velocity = self.forward * v1
                s.transform.position = self.base_point
                s.transform.rotation = lgsvl.Vector(0, 270, 0)
                self.ego.state = s
                state = lgsvl.AgentState()
                state.transform.position = lgsvl.Vector(x2, 0, y2)
                state.transform.rotation = lgsvl.Vector(0, 270, 0)

                npc2 = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)
                waypoints = []
                loc = lgsvl.Vector(x2 - 50, 0, y2 - 1.75)
                wp = lgsvl.DriveWaypoint(
                    position=loc, speed=v2, angle=lgsvl.Vector(0, 270, 0)
                )

                wp1 = lgsvl.DriveWaypoint(
                    position=lgsvl.Vector(x2 - 100, 0, y2 - 3.5), speed=v2, angle=lgsvl.Vector(0, 270, 0)
                )

                wp2 = lgsvl.DriveWaypoint(
                    position=lgsvl.Vector(x2 - 500, 0, y2 - 3.5), speed=v2, angle=lgsvl.Vector(0, 270, 0)
                )

                waypoints.append(wp)
                waypoints.append(wp1)
                waypoints.append(wp2)
                npc2.follow(waypoints)
                self.car_npc['npc2'] = npc2
                state = lgsvl.AgentState()
                state.transform.position = lgsvl.Vector(x3, 0, y3)
                state.transform.rotation = lgsvl.Vector(0, 270, 0)
                state.velocity = self.forward * v3
                npc3 = self.sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)

                self.car_npc['npc3'] = npc3

                vehicles = {
                    self.ego: "EGO",
                    npc2: "Sedan",
                    npc3: "Sedan",
                }

                def on_collision(agent1, agent2, contact):
                    name1 = vehicles[agent1]
                    name2 = vehicles[agent2] if agent2 is not None else "OBSTACLE"
                    if name2 == 'Sedan':
                        # print("{} collided with {} at {}".format(name1, name2, contact))
                        result[-1] = 1
                        # result = 1

                self.ego.on_collision(on_collision)
                # run_simulation(self.sim)
                TTC = 999999
                for i in range(0, 50):
                    ttc_temp = run_simulation(self.sim, self.ego_uid)
                    if ttc_temp < TTC:
                        TTC = ttc_temp
                # 如果没有发生碰撞，result里替换为TTC
                if result[-1] == 0:
                    result[-1] = - TTC

                print(result[-1])

            # s = self.ego.state
            # s.velocity = self.forward * 0
            # self.ego.state = s
            c = lgsvl.VehicleControl()
            c.throttle = 0
            c.braking = 0
            c.steering = 0
            self.ego.apply_control(c, False)
            self.sim_num += 1

        return result




