
import numpy as np
from scipy.optimize import minimize


# 计算正向运动学
def forward_kinematics(dh_params,theta):
    T = np.eye(4)
    theta_diff=np.array([0, -1.43, 1.40, 3.17, np.pi/2,0])
    num_joints=len(dh_params['a'])
    for i in range(num_joints):
        a = dh_params['a'][i]
        alpha = dh_params['alpha'][i]
        d = dh_params['d'][i]
        theta_i = theta[i]+theta_diff[i]
        
        T_i = np.array([
            [np.cos(theta_i),               -np.sin(theta_i),           0,                  a],
            [np.sin(theta_i)*np.cos(alpha),np.cos(theta_i)*np.cos(alpha),-np.sin(alpha),    -d*np.sin(alpha)],
            [np.sin(theta_i)*np.sin(alpha),np.cos(theta_i)*np.sin(alpha),np.cos(alpha), d*np.cos(alpha)],
            [0, 0, 0, 1]
        ])
        
        T = T @ T_i
    return T

#逆运动学目标函数
def inverse_kinematics_target(thetas, target_pos, target_rot, dh_params):
    T = forward_kinematics(dh_params, thetas)
    
    # 末端执行器的位置差异
    end_effector_pos = T[:3, 3]
    position_error = end_effector_pos - target_pos
    
    # 末端执行器的旋转矩阵
    end_effector_rot = T[:3, :3]
    rotation_error = np.dot(end_effector_rot.T, target_rot)  # 目标旋转矩阵与当前旋转矩阵的差异
    
    # 将位置差异和旋转误差合并成一个目标向量
    # 旋转误差转为欧拉角
    rotation_error_euler = np.array([np.arctan2(rotation_error[2, 1], rotation_error[2, 2]),
                                     np.arctan2(-rotation_error[2, 0], np.sqrt(rotation_error[2, 1]**2 + rotation_error[2, 2]**2)),
                                     np.arctan2(rotation_error[1, 0], rotation_error[0, 0])])
    
    # 合并位置误差和旋转误差
    # return np.concatenate([position_error, rotation_error_euler])
    return np.concatenate([position_error])

# 约束条件：关节角度范围
def constraint_bounds(thetas, joint_limits):
    # 约束关节角度在指定范围内
    return np.concatenate([
        thetas - np.array(joint_limits['min']),
        np.array(joint_limits['max']) - thetas
    ])
#逆运动学优化函数
def inverse_kinematics(initial_angle,target_position,joint_limits):
    target_rotation = np.eye(3)  # 假设目标旋转为单位矩阵
    result = minimize(
        fun=lambda thetas: np.sum(inverse_kinematics_target(thetas, target_position, target_rotation, dh_params)**2),
        x0=initial_angle,  # 迭代初始值
        constraints={'type': 'ineq', 'fun': lambda thetas: constraint_bounds(thetas, joint_limits)},  # 角度约束
        method='SLSQP',  # 使用顺序二次规划法（SLSQP）
        options={'disp': True}
    )
    return result.x

if __name__ == "__main__":
    #DH参数
    dh_params = {
    'a': [0, 0, 109.4, 100.51, 0, 8.8],
    'alpha': [0, -np.pi/2, 0, 0, np.pi/2, -np.pi/2],
    'd': [56.3, 22, -5, -17, 58, 20]
    }

    # 关节角度限制
    joint_limits = {
        'min': [-np.pi/2, -2*np.pi/9, -np.pi/2, -np.pi, -np.pi,0],
        'max': [np.pi/2, np.pi/2, np.pi/2, 0, np.pi, 5*np.pi/18]
    }
    
    #正运动学
    angle = np.array([0, 0, 0, 0, 0, 0])
    #angle = np.array([0.1118,-0.2657,0.51828,-0.8983,0.7132,0])
    T_end_effector = forward_kinematics(dh_params,angle)
    print("正运动学:\n", T_end_effector)
    
    #逆运动学
    backward_result=inverse_kinematics(initial_angle = np.zeros(6),target_position=np.array([135.9, 8.8, 109]),joint_limits=joint_limits)
    print("解得的关节角度为(rad)：", backward_result)
    # 将角度转换为度，并保留两位小数
    formatted_result = [round(180 * x / np.pi, 2) for x in backward_result]
    print("解得的关节角度为(degree)：", formatted_result)
    backward_result[2]*= -1
    backward_result[3]*= -1
    backward_result[4]*= -1
    formatted_result = [round(180 * x / np.pi, 2) for x in backward_result]
    print("修正角度为(degree)：", formatted_result)

    control_result = [int((round(180 * x / np.pi, 10)+180)*4096/360) for x in backward_result]
    print("控制量为：", control_result)

