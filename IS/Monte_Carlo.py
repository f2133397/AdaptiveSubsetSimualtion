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

import time
from func_timeout import func_set_timeout
import func_timeout
from goto import with_goto

def test_apolo(self, scenario, env, sim: lgsvl.Simulator, BRIDGE_HOST, BRIDGE_PORT):
    sim_num = 0  # 仿真次数
    base_point = lgsvl.Vector(496, 0, 181.7)  # 本车的初始位置
    car_npc = {}
    spawns = sim.get_spawn()
    # print(spawns[0])
    state = lgsvl.AgentState()
    state.transform.position = base_point
    state.transform.rotation = spawns[0].rotation
    # print(state.transform.position)
    forward = lgsvl.utils.transform_to_forward(spawns[0])  # 获取到前向的方向角
    # state.velocity = v0 * forward
    ego = sim.add_agent(
        env.str("LGSVL__VEHICLE_0", lgsvl.wise.DefaultAssets.ego_lincoln2017mkz_apollo6_modular),
        lgsvl.AgentType.EGO, state)
    ego.connect_bridge(BRIDGE_HOST, BRIDGE_PORT)  # 加入本车并且完成桥接
    # Dreamview setup
    dv = lgsvl.dreamview.Connection(sim, ego, BRIDGE_HOST)

    dv.set_destination(-500, 181.7, 0)

    modules = [
        'Localization',
        'Perception',
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

    for i in range(1, scenario.shape[0]):  # 分别对每条参数进行仿真并且输出仿真结果

        if sim_num > 0:
            sim.remove_agent(car_npc['npc2'])
            del car_npc['npc2']
            sim.remove_agent(car_npc['npc3'])
            del car_npc['npc3']

        # 以下计算各车参数
        v3 = scenario[i][2]
        v2 = v3 - scenario[i][1]
        v1 = v2 - scenario[i][0]
        x2 = base_point.z + scenario[i][4]
        y2 = base_point.x - scenario[i][3] - 5  # 根据车身长度偏移
        x3 = base_point.z
        y3 = y2 - scenario[i][5] - 5  # 根据车身长度偏移

        s = ego.state
        s.velocity = forward * v1
        s.transform.position = base_point
        s.transform.rotation = spawns[0].rotation
        ego.state = s
        state = lgsvl.AgentState()
        state.transform.position = lgsvl.Vector(y2, 0, x2)
        state.transform.rotation = spawns[0].rotation
        # print(v2)
        # print(rotation)
        npc2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)
        waypoints = []
        loc = lgsvl.Vector(y2 - 50, 0, x3)
        wp = lgsvl.DriveWaypoint(
            position=loc, speed=v2, angle=spawns[0].rotation
        )

        wp1 = lgsvl.DriveWaypoint(
            position=lgsvl.Vector(y2 - 100, 0, x3), speed=v2, angle=spawns[0].rotation
        )

        waypoints.append(wp)
        waypoints.append(wp1)
        npc2.follow(waypoints, waypoints_path_type="BezierSpline")  # waypoints_path_type="BezierSpline"

        # npc2.change_lane(True)
        print(sim.available_npc_behaviours)
        # npc2.follow_closest_lane(True,v2,True)
        car_npc['npc2'] = npc2

        # print(npc2.transform)

        state = lgsvl.AgentState()
        state.transform.position = lgsvl.Vector(y3, 0, x3)
        state.transform.rotation = spawns[0].rotation
        state.velocity = forward * v3
        # print(rotation)
        npc3 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)

        car_npc['npc3'] = npc3

        # print(npc3.transform)

        vehicles = {
            ego: "EGO",
            npc2: "Sedan",
            npc3: "sedan",
        }

        # This function gets called whenever any of the 2 vehicles above collides with anything
        def on_collision(agent1, agent2, contact):
            name1 = vehicles[agent1]
            name2 = vehicles[agent2] if agent2 is not None else "OBSTACLE"
            print("{} collided with {} at {}".format(name1, name2, contact))
            scenario[i][6] = 1

        ego.on_collision(on_collision)
        print(ego.get_sensors())
        sim.run(time_limit=5)
        sim_num += 1

    result = pd.DataFrame(scenario)

    result.to_csv("test_result.csv", index=False, sep=',')

@func_set_timeout(10)
def run_simulation(sim: lgsvl.Simulator):
    sim.run(5)

#@with_goto
def test_apolo_gmm( gmm:GaussianMixture(),scaler:StandardScaler(), result,env, sim: lgsvl.Simulator, BRIDGE_HOST, BRIDGE_PORT):
    """
    :param gmm: 高斯混合模型
    :param scaler: 标准化
    :param result：测试结果
    :param env: ENV
    :param sim: 仿真对象
    :param BRIDGE_HOST:  桥接地址
    :param BRIDGE_PORT:  桥接端口
    """

    #label .begin
    sim_num = 0  # 仿真次数
    base_point = lgsvl.Vector(1500, 0, -17.9)  # 本车的初始位置
    car_npc = {}
    state = lgsvl.AgentState()
    state.transform.position = base_point
    state.transform.rotation = lgsvl.Vector(0,270,0)
    forward = lgsvl.utils.transform_to_forward(state)  # 获取到前向的方向角
    state.velocity = 10 * forward

    result = np.array([])
    parameters = np.array([])
    ego = sim.add_agent(
        env.str("LGSVL__VEHICLE_0", "617ff042-1c37-4ff5-af08-598fd1057a26"),
        lgsvl.AgentType.EGO, state)
    ego.connect_bridge(BRIDGE_HOST, BRIDGE_PORT)  # 加入本车并且完成桥接
    # Dreamview setup
    dv = lgsvl.dreamview.Connection(sim, ego, BRIDGE_HOST)

    dv.set_destination(-1500, -17.9, 0)

    modules = [
        'Localization',
        'Perception',
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


    while True:

        #gmm生成1个样本
        scenario = gmm.sample(1)[0]
        scenario = scaler.inverse_transform(scenario)
        #result末尾添加一个0元素
        result = np.append(result, 0)



    #for i in range(1, scenario.shape[0]):  # 分别对每条参数进行仿真并且输出仿真结果

        if sim_num > 0:
            sim.remove_agent(car_npc['npc2'])
            del car_npc['npc2']
            sim.remove_agent(car_npc['npc3'])
            del car_npc['npc3']

        # 以下计算各车参数
        # v3 = scenario[i][2]
        # v2 = v3 - scenario[i][1]
        # v1 = v2 - scenario[i][0]
        # x2 = base_point.z + scenario[i][4]
        # y2 = base_point.x - scenario[i][3] - 5  # 根据车身长度偏移
        # x3 = base_point.z
        # y3 = y2 - scenario[i][5] - 5  # 根据车身长度偏移
        v3 = scenario[0][2]
        v2 = scenario[0][1]
        v1 = scenario[0][0]
        x2 =  base_point.x - scenario[0][3] - 5
        y2 = base_point.z + scenario[0][5]  # 根据车身长度偏移
        x3 = x2 - scenario[0][4] - 5
        y3 = base_point.z

        #输出(scenario[0])到屏幕：当前场景参数为scenario[0]
        print("当前场景参数："+str(scenario[0]))
        parameters = np.append(parameters, scenario[0])

        #print(“场景参数为“+str(scenario[0]))

        s = ego.state
        s.velocity = forward * v1
        s.transform.position = base_point
        s.transform.rotation = lgsvl.Vector(0, 270, 0)
        ego.state = s
        state = lgsvl.AgentState()
        state.transform.position = lgsvl.Vector(x2, 0, y2)
        state.transform.rotation = lgsvl.Vector(0, 270, 0)
        # print(v2)
        # print(rotation)
        npc2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)
        waypoints = []
        loc = lgsvl.Vector(x2 - 50, 0, y2-1.75)
        wp = lgsvl.DriveWaypoint(
            position=loc, speed=v2, angle=lgsvl.Vector(0, 270, 0)
        )

        wp1 = lgsvl.DriveWaypoint(
            position=lgsvl.Vector(x2 - 100, 0, y2-3.5), speed=v2, angle=lgsvl.Vector(0, 270, 0)
        )

        wp2 = lgsvl.DriveWaypoint(
            position=lgsvl.Vector(x2 - 500, 0, y2 - 3.5), speed=v2, angle=lgsvl.Vector(0, 270, 0)
        )

        waypoints.append(wp)
        waypoints.append(wp1)
        waypoints.append(wp2)
        #npc2.follow(waypoints, waypoints_path_type="BezierSpline")  # waypoints_path_type="BezierSpline"
        npc2.follow(waypoints)
        # npc2.change_lane(True)
        #print(sim.available_npc_behaviours)
        # npc2.follow_closest_lane(True,v2,True)
        car_npc['npc2'] = npc2

        # print(npc2.transform)

        state = lgsvl.AgentState()
        state.transform.position = lgsvl.Vector(x3, 0, y3)
        state.transform.rotation = lgsvl.Vector(0, 270, 0)
        state.velocity = forward * v3
        # print(rotation)
        npc3 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)

        car_npc['npc3'] = npc3

        # print(npc3.transform)

        vehicles = {
            ego: "EGO",
            npc2: "Sedan",
            npc3: "sedan",
        }

        # This function gets called whenever any of the 2 vehicles above collides with anything
        def on_collision(agent1, agent2, contact):
            name1 = vehicles[agent1]
            name2 = vehicles[agent2] if agent2 is not None else "OBSTACLE"
            print("{} collided with {} at {}".format(name1, name2, contact))
            #result末尾的0元素替换为1
            result[-1] = 1


        ego.on_collision(on_collision)
        #print(ego.get_sensors())
        try:
            run_simulation(sim)
        except func_timeout.exceptions.FunctionTimedOut as e:
            print(e)
            print("Time out!!!")
            print("请重新开启svl")
            name = input("请输入")
            sim = lgsvl.Simulator(SIMULATOR_HOST, SIMULATOR_PORT)
            if sim.current_scene_id == "085d664e-f3f4-4f38-b303-d45c25125297":
                sim.reset()  # 重置仿真
            else:
                # sim.load("BorregasAve")
                sim.load("085d664e-f3f4-4f38-b303-d45c25125297")


            car_npc = {}
            state = lgsvl.AgentState()
            state.transform.position = base_point
            state.transform.rotation = lgsvl.Vector(0, 270, 0)
            forward = lgsvl.utils.transform_to_forward(state)  # 获取到前向的方向角
            state.velocity = 10 * forward
            ego = sim.add_agent(
                env.str("LGSVL__VEHICLE_0", "617ff042-1c37-4ff5-af08-598fd1057a26"),
                lgsvl.AgentType.EGO, state)
            ego.connect_bridge(BRIDGE_HOST, BRIDGE_PORT)  # 加入本车并且完成桥接
            # Dreamview setup
            dv = lgsvl.dreamview.Connection(sim, ego, BRIDGE_HOST)

            dv.set_destination(-1500, -17.9, 0)

            modules = [
                'Localization',
                'Perception',
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
            #goto .begin
            s = ego.state
            s.velocity = forward * v1
            s.transform.position = base_point
            s.transform.rotation = lgsvl.Vector(0, 270, 0)
            ego.state = s
            state = lgsvl.AgentState()
            state.transform.position = lgsvl.Vector(x2, 0, y2)
            state.transform.rotation = lgsvl.Vector(0, 270, 0)
            # print(v2)
            # print(rotation)
            npc2 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)
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
            # npc2.follow(waypoints, waypoints_path_type="BezierSpline")  # waypoints_path_type="BezierSpline"
            npc2.follow(waypoints)
            # npc2.change_lane(True)
            # print(sim.available_npc_behaviours)
            # npc2.follow_closest_lane(True,v2,True)
            car_npc['npc2'] = npc2

            # print(npc2.transform)

            state = lgsvl.AgentState()
            state.transform.position = lgsvl.Vector(x3, 0, y3)
            state.transform.rotation = lgsvl.Vector(0, 270, 0)
            state.velocity = forward * v3
            # print(rotation)
            npc3 = sim.add_agent("Sedan", lgsvl.AgentType.NPC, state)

            car_npc['npc3'] = npc3

            # print(npc3.transform)

            vehicles = {
                ego: "EGO",
                npc2: "Sedan",
                npc3: "sedan",
            }

            # This function gets called whenever any of the 2 vehicles above collides with anything
            def on_collision(agent1, agent2, contact):
                name1 = vehicles[agent1]
                name2 = vehicles[agent2] if agent2 is not None else "OBSTACLE"
                print("{} collided with {} at {}".format(name1, name2, contact))
                # result末尾的0元素替换为1
                result[-1] = 1

            ego.on_collision(on_collision)
            # print(ego.get_sensors())
            run_simulation(sim)



        s= ego.state
        s.velocity = forward * 0
        ego.state = s
        c = lgsvl.VehicleControl()
        c.throttle = 0
        c.braking = 0
        ego.apply_control(c, False)
        sim_num += 1

        rate,half_width = cal_crash_and_width(result)
        if half_width < 0.2:
            print("half_width:", half_width)
            print("rate:", rate)
            #result输出到csv文件
            np.savetxt('result_1229.csv', result, delimiter=',')
            #parameters输出到csv文件
            np.savetxt('parameters_1229.csv', parameters, delimiter=',')
            break
        else:
            print("half_width:",half_width)
            print("rate:",rate)
        #result = pd.DataFrame(scenario)

        #result.to_csv("test_result.csv", index=False, sep=',')

def cal_crash_and_width(result, z_rate=0.95):
    """计算事故率以及相对半宽al

    :param result: 测试环境得到的测试结果。注意，对于重要度抽样方法计算得到的result，需要先抵消不同分布带来的误差。即 测试结果 * p(x) / q(x)，再传入本方法中。
    :param z_rate: 置信度
    :return:
        rate,half_width
        事故率与相对半宽
    """
    z = ndtri(1 - (1 - z_rate) / 2)

    rate = result.mean()
    if rate != 0:
        # 置信区间为 rate + ndtri(0.9)*S/n
        # l_r = z * np.sqrt((1 - rate) / rate / result.shape[0])
        l_r = z * result.std() / np.sqrt(result.shape[0]) / rate
    else:
        l_r = 1

    return rate, l_r

if __name__ == "__main__":
    env = Env()
    SIMULATOR_HOST = os.environ.get("SIMULATOR_HOST", "127.0.0.1")
    SIMULATOR_PORT = int(os.environ.get("SIMULATOR_PORT", 8181))
    BRIDGE_HOST = os.environ.get("BRIDGE_HOST", "127.0.0.1")
    BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", 9090))

    data = pd.read_csv("cutin_diswithin50.csv", sep=',')
    #获取列名list
    #col = data.columns.tolist()
    #取v1，v2，v3，dis1x,dis2x,dis1y所在的列
    data = data.loc[:,['v1','v2','v3','dis1x','dis2x','dis1y']]
    result = np.array([])
    s=1
    # 拟合归一化工具，对数据进行归一化，能够使得拟合更加准确
    scaler = StandardScaler()
    scaler.fit(data)
    # 为了提高拟合效率，只使用部分数据进行高斯混合模型组分数量确定以及拟合的工作
    data_trans = scaler.transform(data)

    # 通过bic指数，确定最适合的高斯混合模型组分数量
    best_components = 1
    best_bic = 999999999
    for i in range(15):
        gmm = GaussianMixture(n_components=i + 1, tol=0.0001, max_iter=10000)
        gmm.fit(data_trans)
        bic = gmm.bic(data_trans)
        if bic < best_bic:
            best_bic = bic
            best_components = i + 1
    # 确定完毕最合适的高斯混合模型组分数量后，使用部分数据拟合高斯混合模型
    best_gmm = GaussianMixture(n_components=best_components, tol=0.0001, max_iter=10000)
    best_gmm.fit(data_trans)
    #从best_gmm中随机采样出1000条数据
    data_sample = best_gmm.sample(1)[0]
    # 将采样出的数据进行反归一化，得到真实的数据
    data_sample = scaler.inverse_transform(data_sample)

    sim = lgsvl.Simulator(SIMULATOR_HOST, SIMULATOR_PORT)
    if sim.current_scene_id == "085d664e-f3f4-4f38-b303-d45c25125297":
        sim.reset()  # 重置仿真
    else:
        # sim.load("BorregasAve")
        sim.load("085d664e-f3f4-4f38-b303-d45c25125297")
    test_apolo_gmm(best_gmm,scaler,result,env, sim, BRIDGE_HOST, BRIDGE_PORT)

