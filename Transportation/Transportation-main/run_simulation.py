import os
import sys
import traci

# --- 步骤 1: 设置 SUMO_HOME ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# --- 步骤 2: 定义 SUMO 启动命令 ---
sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui.exe')
sumoCmd = [sumoBinary, "-c", "test.sumocfg"]

# --- 步骤 3: 启动 Traci ---
print("Starting SUMO simulation...")
traci.start(sumoCmd)

# --- 步骤 4: 运行仿真主循环 ---
step = 0
hdv_count = 0
av_count = 0

while traci.simulation.getMinExpectedNumber() > 0:
    # 1. 仿真步进
    traci.simulationStep()

    # 2. 获取新出发车辆
    departed_vehicles = traci.simulation.getDepartedIDList()

    # 3. 统计车辆类型 (使用 'in' 模糊匹配)
    for v_id in departed_vehicles:
        v_type = traci.vehicle.getTypeID(v_id)

        if "HDV" in v_type:
            hdv_count += 1
        elif "AV" in v_type:
            av_count += 1

    # 4. 定期打印进度
    if step % 100 == 0:
        print(f"Step {step}: Total HDV={hdv_count}, Total AV={av_count}")

    step += 1

# --- 步骤 5: 结束 ---
traci.close()
print("Simulation ended.")
print(f"Final Count: HDV={hdv_count}, AV={av_count}")