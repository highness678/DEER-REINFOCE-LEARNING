import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
import os

# 导入你写好的环境类
from env_sumo import SumoAVEnv

print("【调试 1/4】程序已启动，正在准备环境...") 

def make_env(rank, seed=0):
    """
    环境工厂函数：用于创建独立的环境实例。
    
    Args:
        rank (int): 进程的索引 (0, 1, 2, ...)
        seed (int): 随机种子
    """
    def _init():
        # 1. 实例化环境
        # 注意：并行训练时必须关闭 GUI (use_gui=False)
        # 这里的 cfg 路径建议写绝对路径，或者确保相对路径正确
        print("【调试 2/4】正在尝试启动 SUMO 仿真器，请留意任务栏有没有新窗口...")
        env = SumoAVEnv(
            sumo_cfg_path="test.sumocfg", 
            use_gui=False,
            step_length=0.1,
            control_dt=0.5,
            max_steps=3600
        )
        
        # 2. 设置随机种子 (让每个环境的随机性略有不同，增加探索性)
        env.reset(seed=seed + rank)
        return env
        
    return _init

print("【调试 3/4】SUMO 启动成功！准备开始训练循环...")

if __name__ == "__main__":
    # ==========================================
    # 👇 调试模式修改开始
    # ==========================================
    
    # 1. 暂时只用 1 个 CPU，方便看报错
    num_cpu = 1 
    print(f"【调试模式】正在启动 {num_cpu} 个 SUMO 环境...")

    # 2. 修改 make_env 里的参数用于调试
    # 我们手动创建一个临时的工厂函数，强制开启 GUI
    def make_debug_env(rank, seed=0):
        def _init():
            print("【调试】正在启动带界面的 SUMO...")
            # ⚠️ 注意：这里把 use_gui 改成了 True，让你能看到画面！
            env = SumoAVEnv(
                sumo_cfg_path="test.sumocfg", 
                use_gui=True,  
                step_length=0.1,
                control_dt=0.5,
                max_steps=3600
            )
            return env
        return _init

    # 3. 使用 DummyVecEnv (单线程)，而不是 SubprocVecEnv (多进程)
    # 这样报错会直接显示在终端，不会被隐藏
    env = DummyVecEnv([make_debug_env(0)])
    
    env = VecMonitor(env, filename="./logs/monitor_logs")

    # 4. 定义模型 - 关键修改！
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./logs/tensorboard",
        learning_rate=3e-4,
        batch_size=64,   # 改小
        n_steps=128,     # ⚠️ 极其重要：改小！
                         # 只要跑 128 步就会打印一次日志，你会立刻看到反应
        device="auto"
    )

    # 5. 开始训练
    print("【调试】开始训练... 请留意弹出的 SUMO 窗口，并点击播放(Play)！")
    model.learn(total_timesteps=10000)

    model.save("ppo_sumo_debug")
    print("调试完成。")
    env.close()
