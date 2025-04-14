import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random
import tkinter as tk
from tkinter import simpledialog, scrolledtext
import threading

link_lengths = [0.333, 0.316, 0.0825, 0.0825, 0.088, 0.085, 0.1]

joint_ranges = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973),
                (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]

def calculate_inverse_kinematics(robot_id, target_position):
    return p.calculateInverseKinematics(robot_id, 11, target_position)

def is_valid_target_position(robot_id, target_position):
    x, y, z = target_position
    log(f"\n🔍 Kontrola bodu: {target_position}")

    if np.linalg.norm([x, y]) < 0.05 or z < 0:
        log("❌ Neplatná pozícia! Bod je príliš blízko základne alebo pod podlahou.")
        return False

    max_reach = 0.855
    if np.sqrt(x**2 + y**2) > max_reach or z > 1.190:
        log(f"❌ Bod je mimo pracovného rozsahu! Vypočítaná horizontálna vzdialenosť: {np.sqrt(x**2 + y**2)}, Z = {z}")
        return False

    try:
        joint_angles = calculate_inverse_kinematics(robot_id, target_position)
        log(f"🔄 Vypočítané uhly kĺbov: {joint_angles}")

        for i in range(7):
            if not (joint_ranges[i][0] <= joint_angles[i] <= joint_ranges[i][1]):
                log(f"❌ Kĺb {i+1} mimo rozsahu! ({joint_angles[i]} nie je v {joint_ranges[i]})")
                return False
    except Exception as e:
        log(f"⚠️ Chyba pri výpočte IK: {e}")
        return False

    log("✅ Pozícia je validná!")
    return True

def visualize_workspace():
    points = []
    for theta in np.linspace(0, 2 * np.pi, 20):
        for z in np.linspace(0, 1.19, 15):
            max_reach = 0.855
            r = max_reach * np.sqrt(1 - (z / 1.19)**3)
            x, y = r * np.cos(theta), r * np.sin(theta)
            points.append([x, y, z])
    for point in points:
        p.addUserDebugPoints([point], [[0, 0, 1]], pointSize=3)
    return np.array(points)

def sample_workspace_points(robot_id, num_samples=5000, tolerance=0.01):
    positions = []
    revolute_joint_indices = []

    for joint_index in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, joint_index)
        if joint_info[2] == p.JOINT_REVOLUTE:
            revolute_joint_indices.append(joint_index)

    for _ in range(num_samples):
        joint_values = [random.uniform(joint_ranges[i][0], joint_ranges[i][1]) for i in range(7)]
        for idx, val in zip(revolute_joint_indices, joint_values):
            p.resetJointState(robot_id, idx, val)

        true_state = p.getLinkState(robot_id, 11)
        true_pos = np.array(true_state[4])

        if true_pos[2] < 0:
            continue

        ik_solution = calculate_inverse_kinematics(robot_id, true_pos)

        for i in range(7):
            p.resetJointState(robot_id, revolute_joint_indices[i], ik_solution[i])

        confirmed_state = p.getLinkState(robot_id, 11)
        confirmed_pos = np.array(confirmed_state[4])

        if np.linalg.norm(true_pos - confirmed_pos) < tolerance:
            positions.append(true_pos)

    p.addUserDebugPoints(
        pointPositions=positions,
        pointColorsRGB=[[0, 0, 1]] * len(positions),
        pointSize=2
    )

    return np.array(positions)

def visualize_target_point(target_position):
    p.addUserDebugPoints([target_position], [[1, 0, 0]], pointSize=10)

def plot_workspace_and_target(target_position, workspace_points):
    dist = distance.cdist([target_position], workspace_points)
    nearest_point_idx = np.argmin(dist)
    nearest_point = workspace_points[nearest_point_idx]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2], c='blue', s=1.5, alpha=0.6)
    ax.scatter(target_position[0], target_position[1], target_position[2], c='red', s=50)
    ax.scatter(nearest_point[0], nearest_point[1], nearest_point[2], c='green', s=50)
    ax.plot([target_position[0], nearest_point[0]], [target_position[1], nearest_point[1]], [target_position[2], nearest_point[2]], color='black', linewidth=2)

    ax.text(target_position[0], target_position[1], target_position[2],
            f'[{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]',
            color='red', fontsize=10)
    ax.text(nearest_point[0], nearest_point[1], nearest_point[2],
            f'[{nearest_point[0]:.3f}, {nearest_point[1]:.3f}, {nearest_point[2]:.3f}]',
            color='green', fontsize=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Pracovný priestor robota')
    plt.show()

def move_to_home(robot_id):
    home_angles = [0, -np.pi/4, 0, -np.pi/2, 0, np.pi/2, 0]
    for i in range(7):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=home_angles[i], force=100)
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(1/400)

def move_to_target(robot_id, joint_angles):
    for i in range(7):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_angles[i], force=100)
    for _ in range(1000):
        p.stepSimulation()
        time.sleep(1/400)

def log(msg):
    output_text.insert(tk.END, msg + "\n")
    output_text.see(tk.END)

def open_input_dialog():
    input_str = simpledialog.askstring("Zadajte cieľ", "Zadajte X Y Z (v mm) napr. 100 200 300")
    if input_str:
        try:
            x_mm, y_mm, z_mm = map(float, input_str.split())
            target_position = [x_mm / 1000, y_mm / 1000, z_mm / 1000]
            log(f"➡️ Zadaný cieľ: {target_position}")

            if not is_valid_target_position(robot_id, target_position):
                plot_workspace_and_target(target_position, workspace_points)
                return

            visualize_target_point(target_position)
            joint_angles = calculate_inverse_kinematics(robot_id, target_position)
            move_to_target(robot_id, joint_angles)
            log(f"💡 Robot dosiahol cieľový bod: {target_position}")

        except ValueError:
            log("❌ Zadaný formát nie je platný. Skúste to znova.")

def simulation_thread():
    while True:
        p.stepSimulation()
        time.sleep(1/240)

if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

    log_root = tk.Tk()
    log_root.title("Robot GUI")
    output_text = scrolledtext.ScrolledText(log_root, wrap=tk.WORD, width=80, height=20)
    output_text.pack()

    button = tk.Button(log_root, text="Zadať cieľový bod", command=open_input_dialog)
    button.pack()

    home_button = tk.Button(log_root, text="⬅️ Vrátiť do Home", command=lambda: move_to_home(robot_id))
    home_button.pack()

    log("🧠 Vzorkujem reálny pracovný priestor robota...")
    workspace_points = sample_workspace_points(robot_id, num_samples=3000)
    log("✅ Realistický pracovný priestor bol úspešne vizualizovaný.")

    move_to_home(robot_id)
    log("📌 Robot nastavený do počiatočnej pozície.")

    threading.Thread(target=simulation_thread, daemon=True).start()
    log_root.mainloop()
