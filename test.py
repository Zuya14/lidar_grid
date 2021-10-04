from square3Env import square3Env
from maze3Env import maze3Env
import math
# env = square3Env()
env = maze3Env()
env.setting()

# x, y = env.observe_all()[0][:2]
# print(x,y)

from lidar_to_grid_map import generate_ray_casting_grid_map
import numpy as np
import matplotlib.pyplot as plt
from GridScan import GridScan

plt.figure()

while True:
    action = env.sample_random_action()

    state, _, done, _ = env.step(action)

    scan = GridScan(lidar_maxLen=10.0, resolution=0.2)

    rads, dist = env.scan()
    ox = np.sin(rads) * dist
    oy = np.cos(rads) * dist
    occupancy_map = scan.generate_ray_casting_grid_map(ox, oy)
    plt.imshow(occupancy_map, cmap="PiYG_r")
    plt.pause(0.1)

    if done:
        break



# occupancy_map, min_x, max_x, min_y, max_y, xy_resolution = \
#     generate_ray_casting_grid_map(ox, oy, xy_resolution, True)

# print(occupancy_map, min_x, max_x, min_y, max_y, xy_resolution)

# # xy_res = np.array(occupancy_map).shape
# plt.figure(1, figsize=(10, 4))
# plt.figure()

# plt.subplot(122)
# plt.imshow(occupancy_map, cmap="PiYG_r")
# # cmap = "binary" "PiYG_r" "PiYG_r" "bone" "bone_r" "RdYlGn_r"
# # # plt.clim(-0.4, 1.4)
# # # plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=True)
# # # plt.gca().set_xticks(np.arange(-.5, xy_res[1], 1), minor=False)
# # # plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=True)
# # # plt.gca().set_yticks(np.arange(-.5, xy_res[0], 1), minor=False)
# # # plt.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
# # # plt.grid(True, color="w", linewidth=0.6, alpha=0.5)
# # plt.colorbar()

# # plt.subplot(121)
# # plt.plot([ox+x, np.full(np.size(oy), x)], [oy+y, np.full(np.size(oy), y)], "ro-")
# # plt.axis("equal")
# # plt.plot(x, y, "ob")
# # # plt.gca().set_aspect("equal", "box")
# # # bottom, top = plt.ylim()  # return the current y-lim
# # # plt.ylim((top, bottom))  # rescale y axis, to match the grid orientation
# # # plt.ylim((bottom, top))  # rescale y axis, to match the grid orientation
# # plt.grid(True)

# plt.show()
