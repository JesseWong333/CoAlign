# import json 
# import numpy as np
# import math

# '''

#           4 -------- 5
#          /|         /|
#         7 -------- 6 .
#         | |        | |
#         . 0 -------- 1
#         |/         |/
#         3 -------- 2

#                         front z
#                              /
#                             /
#               (x0, y0, z1) + -----------  + (x1, y0, z1)
#                           /|            / |
#                          / |           /  |
#            (x0, y0, z0) + ----------- +   + (x1, y1, z1)
#                         |  /      .   |  /
#                         | / oriign    | /
#            (x0, y1, z0) + ----------- + -------> x right
#                         |             (x1, y1, z0)
#                         |
#                         v
#                    down y

#    """
# '''


# def convert_CuboidRepresentation2vertices(object):
#     # 测试ChatGPT写的坐标转换是否正确; 不一样
#     x = object["3d_location"]["x"]
#     y = object["3d_location"]["y"]
#     z = object["3d_location"]["z"]

#     h = object["3d_dimensions"]["h"]  # h 车高， y
#     w =  object["3d_dimensions"]["w"] # w 车宽， x
#     l = object["3d_dimensions"]["l"]  # l 是车长 z

#     center = np.array([x, y, z])  # 立方体的中心点坐标
#     dimensions = np.array([w, h, z])  # 立方体的长宽高
#     rotation_angle = -object["rotation"]  # 旋转角度（弧度） # 原来给出的是弧度; 是否是旋转方向不对？

#     # 计算立方体的半轴
#     half_lengths = dimensions / 2

#     # 定义立方体的8个顶点相对中心的偏移量
#     vertices_offsets = np.array([
#         [-1, -1, -1],  # 前下左
#         [-1, -1, 1],   # 前下右
#         [-1, 1, -1],   # 前上左
#         [-1, 1, 1],    # 前上右
#         [1, -1, -1],   # 后下左
#         [1, -1, 1],    # 后下右
#         [1, 1, -1],    # 后上左
#         [1, 1, 1]      # 后上右
#     ])

#     # 应用旋转变换, 绕z轴旋转
#     # rotation_matrix = np.array([
#     #     [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
#     #     [np.sin(rotation_angle), np.cos(rotation_angle), 0],
#     #     [0, 0, 1]
#     # ])
#     # 绕y轴旋转
#     rotation_matrix = np.array([
#         [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
#         [0, 1, 0],
#         [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
#     ])

#     rotated_vertices_offsets = np.matmul(rotation_matrix, vertices_offsets.T).T
#     # 计算顶点坐标
#     vertices = center + rotated_vertices_offsets * half_lengths

#     return vertices

# # 这个是官方写的
# def get_3d_8points(object):
#     h = object["3d_dimensions"]["h"] 
#     w = object["3d_dimensions"]["w"] 
#     l = object["3d_dimensions"]["l"] 

#     yaw_lidar = object["rotation"]
#     x = object["3d_location"]["x"]
#     y = object["3d_location"]["y"]
#     z = object["3d_location"]["z"]
#     center_lidar = [x, y, z]
#     liadr_r = np.matrix(
#         [
#             [math.cos(yaw_lidar), -math.sin(yaw_lidar), 0],
#             [math.sin(yaw_lidar), math.cos(yaw_lidar), 0],
#             [0, 0, 1],
#         ]
#     )
   
#     corners_3d_lidar = np.matrix(
#         [
#             [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
#             [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
#             [0, 0, 0, 0, h, h, h, h],
#         ]
#     )
#     corners_3d_lidar = liadr_r * corners_3d_lidar + np.matrix(center_lidar).T

#     return corners_3d_lidar.T


# def read_json(path):
#     with open(path, "r") as f:
#         my_json = json.load(f)
#     return my_json


# if __name__ == '__main__':
#     label_path = "/data/datasets/DAIR-V2X/cooperative-vehicle-infrastructure/cooperative/label_world/000010.json"
#     GT = read_json(label_path)

#     for object in GT:
#         xx = get_3d_8points(object)
#         print(xx)
#         print(np.array(object["world_8_points"]))
#         pass

# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt

# def trans_rotate(x, y, px, py, beta):
#     beta = np.deg2rad(beta)
#     T = [[np.cos(beta), -np.sin(beta), px*(1 - np.cos(beta)) + py*np.sin(beta)],
#          [np.sin(beta),  np.cos(beta), py*(1 - np.cos(beta)) - px*np.sin(beta)],
#          [0           ,  0           , 1                                      ]]
#     T = np.array(T)
#     P = np.array([x, y, [1]*x.size])  # x: [(N, ) y:(N,) [1,1...1]]
#     return np.dot(T, P)

# fig, ax = plt.subplots(1, 4)

# R_ = [0, 225, 40, -10]
# P_ = [[0, 0], [0, 0], [0.5, -0.5], [1.1, 1.1]]

# X, Y = np.mgrid[0:1:5j, 0:1:5j]
# x, y = X.ravel(), Y.ravel()

# for i in range(4):
#     beta = R_[i]; px, py = P_[i]
#     x_, y_, _ = trans_rotate(x, y, px, py, beta)
#     ax[i].scatter(x_, y_)
#     ax[i].scatter(px, py)
#     ax[i].set_title(r'$\beta={0}°$ , $p_x={1:.2f}$ , $p_y={2:.2f}$'.format(beta, px, py))
    
#     ax[i].set_xlim([-2, 2])
#     ax[i].set_ylim([-2, 2])
#     ax[i].grid(alpha=0.5)
#     ax[i].axhline(y=0, color='k')
#     ax[i].axvline(x=0, color='k')

# plt.show()

import numpy as np
for i in range(100):
    print(np.floor(np.random.exponential(scale=2.0)).astype(np.int32))
