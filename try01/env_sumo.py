import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys

# # 确保 SUMO_HOME 存在
# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     # 你的硬编码路径作为备选
#     sys.path.append(r'D:\sumo\sumo-1.21.0\tools')

import traci
import sumolib  

class SumoAVEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 sumo_cfg_path="test.sumocfg",
                 sumo_home=None,
                 use_gui=False,
                 step_length=0.1,
                 control_dt=0.5,
                 max_steps=3600,
                 agent_type_key="AV",
                 main_edge="E1",
                 lane_count=3):
        super().__init__()
        self.sumo_cfg_path = sumo_cfg_path
        self.sumo_home = sumo_home or os.environ.get("SUMO_HOME")
        
        if not self.sumo_home:
            raise RuntimeError("请设置系统环境变量 SUMO_HOME")

        self.use_gui = use_gui
        self.step_length = step_length
        self.k_sim_steps = int(round(control_dt / step_length))
        self.max_steps = int(max_steps / control_dt)
        self.agent_type_key = agent_type_key
        self.main_edge = main_edge
        self.lane_count = lane_count

        # --- 修复点：定义 sumoBinary ---
        if self.use_gui:
            self.sumoBinary = sumolib.checkBinary('sumo-gui')
        else:
            self.sumoBinary = sumolib.checkBinary('sumo')

        # 动作空间：换道 × 速度档
        self.lane_action = [-1, 0, 1]
        self.speed_action = [20.0, 25.0, 30.0]  # m/s
        self.action_space = spaces.MultiDiscrete([len(self.lane_action), len(self.speed_action)])

        # 状态空间
        high = np.array([40.0, 3.0, 300.0, 40.0, 300.0, 40.0, 300.0, 300.0, 2.0, 2.0, 2.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.agent_id = None
        self.current_step = 0

    def _sumo_cmd(self):
        # 获取当前脚本所在的文件夹路径（跨平台兼容）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = os.path.normpath(current_dir)
    
        # 拼接路径：跳出 try01 -> 进入 Transportation-main -> 再进入 Transportation-main
        cfg_path = os.path.join(
            current_dir, 
            "..", 
            "Transportation", 
            "Transportation-main", 
            "test.sumocfg"
        )
        
        # 转成绝对路径
        cfg_abs = os.path.abspath(cfg_path)
        
        # 打印调试信息
        print(f"正在尝试加载配置文件: {cfg_abs}")

        if not os.path.exists(cfg_abs):
            raise FileNotFoundError(f"找不到 SUMO 配置文件，请检查路径: {cfg_abs}")
            
        # 返回命令
        return [self.sumoBinary, "-c", cfg_abs, "--step-length", str(self.step_length)]

    def reset(self, seed=None, options=None):
        # 1. 处理随机种子
        if seed is not None:
        # self.seed(seed) 
         pass

        # 2. 暴力清理旧连接
        try:
            traci.close()
        except Exception:
            pass
            
        import time
        time.sleep(0.1)

        # 3. 启动 SUMO
        traci.start(self._sumo_cmd())
        
        self.current_step = 0
        self.agent_id = None

        # 4. 等待车辆进入 (最多等待 3000 步)
        for _ in range(3000):
            traci.simulationStep()
            deps = traci.simulation.getDepartedIDList()
            for vid in deps:
                vtype = traci.vehicle.getTypeID(vid)
                if self.agent_type_key in vtype:
                    self.agent_id = vid
                    break
            if self.agent_id is not None:
                break
        
        # 5. 检查是否成功找到车
        if self.agent_id is None:
            traci.close()
            raise RuntimeError(f"❌ 错误: 3000步内未找到类型为 {self.agent_type_key} 的车辆，请检查 rou.xml")

        # 6. 获取初始观测
        obs = self._get_obs()
        
        # 7. ✅ 返回 obs 和 info (这是 Gym API 的标准要求)
        return obs, {}




    def step(self, action):
        info = {}
        # 如果智能体不在网络中
        if self.agent_id is None or self.agent_id not in traci.vehicle.getIDList():
            # 检查是否是因为刚刚到达终点
            if self.agent_id and self.agent_id in traci.simulation.getArrivedIDList():
                 reward, comps = self._reward_components(is_arrived_step=True)
                 return np.zeros(self.observation_space.shape, dtype=np.float32), reward, True, False, comps
            
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, info

        # 解析动作
        lane_delta = self.lane_action[int(action[0])]
        speed_cmd = self.speed_action[int(action[1])]

        allowed = traci.vehicle.getAllowedSpeed(self.agent_id)
        speed_cmd = float(min(speed_cmd, allowed))

        # 换道
        try:
            curr_lane = traci.vehicle.getLaneIndex(self.agent_id)
            target_lane = np.clip(curr_lane + lane_delta, 0, self.lane_count - 1)
            if lane_delta != 0 and target_lane != curr_lane:
                # 注意：换道持续时间不能太短，否则 SUMO 会忽略
                traci.vehicle.changeLane(self.agent_id, int(target_lane), 2.0)
        except traci.TraCIException:
            pass

        traci.vehicle.setSpeed(self.agent_id, speed_cmd)

        # 推进仿真
        collision_occurred = False
        for _ in range(self.k_sim_steps):
            traci.simulationStep()
            self.current_step += 1
            if self._check_collision():
                collision_occurred = True
        
        self.last_step_collision = collision_occurred

        # 计算奖励与观测
        reward, comps = self._reward_components()
        obs = self._get_obs()
        done = self._is_done() or collision_occurred
        info.update(comps)
        
        return obs, reward, done, False, info

    # -----------------------------------------------------------
    #  核心部分：数据收集与奖励计算
    # -----------------------------------------------------------

    def _check_collision(self):
        colliding_vehs = traci.simulation.getCollidingVehiclesIDList()
        return self.agent_id in colliding_vehs

    def _get_surrounding_acc(self):
        acc_list = []
        if self.agent_id is None: return acc_list
        
        try:
            leader_info = traci.vehicle.getLeader(self.agent_id)
            if leader_info:
                acc_list.append(traci.vehicle.getAcceleration(leader_info[0]))
                
            follower_info = traci.vehicle.getFollower(self.agent_id)
            if follower_info and follower_info[1] < 50.0:
                acc_list.append(traci.vehicle.getAcceleration(follower_info[0]))
        except Exception:
            pass
            
        return acc_list

    def _reward_components(self, is_arrived_step=False):
        vid = self.agent_id
        comps = {}

        env_info = {
            'is_collision': False,
            'is_success': False,
            'velocity': 0.0,
            'max_speed': 30.0,
            'acceleration': 0.0,
            'min_distance_to_other': 100.0,
            'surrounding_vehicles_acc': []
        }

        if is_arrived_step:
            env_info['is_success'] = True
        elif vid in traci.vehicle.getIDList():
            env_info['is_collision'] = getattr(self, 'last_step_collision', False)
            env_info['is_success'] = False
            env_info['velocity'] = traci.vehicle.getSpeed(vid)
            env_info['max_speed'] = traci.vehicle.getAllowedSpeed(vid)
            env_info['acceleration'] = traci.vehicle.getAcceleration(vid)
            
            dx_f, _ = self._neighbor_rel(vid, ahead=True)
            dx_b, _ = self._neighbor_rel(vid, ahead=False)
            dx_l = self._side_front_dist(vid, left=True)
            dx_r = self._side_front_dist(vid, left=False)
            env_info['min_distance_to_other'] = min(dx_f, dx_b, dx_l, dx_r)
            
            env_info['surrounding_vehicles_acc'] = self._get_surrounding_acc()
        
        elif getattr(self, 'last_step_collision', False):
             env_info['is_collision'] = True
             env_info['min_distance_to_other'] = 0.0

        reward = self._compute_advanced_reward(env_info)
        
        comps['raw_reward'] = reward
        comps['is_collision'] = env_info['is_collision']
        comps['is_success'] = env_info['is_success']
        
        return float(reward), comps

    def _compute_advanced_reward(self, env_info):
        W_COLLISION = -100.0
        W_SUCCESS = 20.0
        W_SPEED = 0.5
        W_TIME = -0.05
        W_JERK = -0.1
        W_SAFETY = -1.5
        W_SOCIAL = -2.0

        reward = 0.0
        
        if env_info['is_collision']:
            return W_COLLISION
        
        if env_info['is_success']:
            return W_SUCCESS

        if env_info['max_speed'] > 0:
            v_ratio = env_info['velocity'] / env_info['max_speed']
            reward += W_SPEED * np.clip(v_ratio, 0, 1)
        reward += W_TIME 

        acc_penalty = abs(env_info['acceleration'])
        if env_info['acceleration'] < -3.0:
            acc_penalty *= 2.0
        reward += W_JERK * acc_penalty

        safe_threshold = 5.0
        dist = env_info['min_distance_to_other']
        if dist < safe_threshold:
            penalty = (safe_threshold - dist) ** 2
            reward += W_SAFETY * penalty

        for hv_acc in env_info['surrounding_vehicles_acc']:
            if hv_acc < -3.5:
                reward += W_SOCIAL
                break 

        return reward

    # -----------------------------------------------------------
    #  辅助函数
    # -----------------------------------------------------------

    def _get_obs(self):
        if self.agent_id is None or self.agent_id not in traci.vehicle.getIDList():
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        vid = self.agent_id
        v = traci.vehicle.getSpeed(vid)
        lane = float(traci.vehicle.getLaneIndex(vid))

        dx_f, dv_f = self._neighbor_rel(vid, ahead=True)
        dx_b, dv_b = self._neighbor_rel(vid, ahead=False)
        dx_left_f = self._side_front_dist(vid, left=True)
        dx_right_f = self._side_front_dist(vid, left=False)

        dens = self._lane_density(self.main_edge)
        dens = (dens + [0, 0, 0])[:3]

        obs = np.array([
            v, lane,
            dx_f, dv_f,
            dx_b, dv_b,
            dx_left_f, dx_right_f,
            float(dens[0]), float(dens[1]), float(dens[2])
        ], dtype=np.float32)
        return obs

    def _neighbor_rel(self, vid, ahead=True):
        lane_id = traci.vehicle.getLaneID(vid)
        pos = traci.vehicle.getLanePosition(vid)
        speed = traci.vehicle.getSpeed(vid)

        leader = traci.vehicle.getLeader(vid)
        follower = traci.vehicle.getFollower(vid)
        neigh_id = leader[0] if ahead and leader else (follower[0] if (not ahead and follower) else None)
        if not neigh_id:
            return 300.0, 0.0

        try:
            if traci.vehicle.getLaneID(neigh_id) != lane_id:
                return 300.0, 0.0
            n_pos = traci.vehicle.getLanePosition(neigh_id)
            n_speed = traci.vehicle.getSpeed(neigh_id)
            dx = (n_pos - pos) if ahead else (pos - n_pos)
            dv = n_speed - speed
            if dx < 0:
                return 300.0, 0.0
            return float(min(dx, 300.0)), float(np.clip(dv, -40.0, 40.0))
        except traci.TraCIException:
            return 300.0, 0.0

    def _side_front_dist(self, vid, left=True):
        try:
            curr_lane = traci.vehicle.getLaneIndex(vid)
            target_lane = curr_lane + (-1 if left else 1)
            if target_lane < 0 or target_lane >= self.lane_count:
                return 300.0
            lane_id = f"{self.main_edge}_{int(target_lane)}"
            vehs = traci.lane.getLastStepVehicleIDs(lane_id)
            pos = traci.vehicle.getLanePosition(vid)
            min_dx = 300.0
            for nv in vehs:
                npos = traci.vehicle.getLanePosition(nv)
                if npos > pos:
                    min_dx = min(min_dx, npos - pos)
            return float(min_dx)
        except traci.TraCIException:
            return 300.0

    def _lane_density(self, edge_id):
        try:
            lanes = traci.edge.getLaneNumber(edge_id)
            dens = []
            for i in range(lanes):
                lane_id = f"{edge_id}_{i}"
                count = traci.lane.getLastStepVehicleNumber(lane_id)
                length = traci.lane.getLength(lane_id)
                window = min(200.0, length)
                dens.append(count / max(1.0, window/200.0))
            return dens
        except traci.TraCIException:
            return []

    def _is_done(self):
        if self.current_step >= self.max_steps:
            return True
        if self.agent_id and self.agent_id in traci.simulation.getArrivedIDList():
            return True
        if traci.simulation.getMinExpectedNumber() == 0:
            return True
        return False

    def render(self):
        pass

    def close(self):
        try:
            if traci.isLoaded():
                traci.close(False)
        except Exception as e:
            print("[SUMO-CLOSE] Warning:", e)

if __name__ == "__main__":
    # 简单的测试代码
    env = SumoAVEnv(use_gui=True)
    obs, _ = env.reset()
    print("Env reset done. Obs:", obs)
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i}: Reward={reward:.2f}, Done={done}")
        if done:
            break
    env.close()
