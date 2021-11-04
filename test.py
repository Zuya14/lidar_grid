from square3Env import square3Env
from maze3Env import maze3Env
import math
from lidar_to_grid_map import generate_ray_casting_grid_map
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from GridScan import GridScan
from NoveltyBuffer import NoveltyBuffer
from CuriosityEnv1 import CuriosityEnv1
from CuriosityEnv2 import CuriosityEnv2
from CuriosityEnv3 import CuriosityEnv3
from CuriosityEnv4 import CuriosityEnv4

def saveMapCsv():
    # env = square3Env()
    env = maze3Env()
    env.setting()

    plt.figure()
    x, y = 0, 0
    occupancy_map = None

    for _ in range(100):
        state = env.reset()
        x, y = state[0], state[1]

        scan = GridScan(lidar_maxLen=10.0, resolution=0.1)

        rads, dist = env.scan()
        ox = np.sin(rads) * dist
        oy = np.cos(rads) * dist

        if occupancy_map is None:
            space_map, obstacle_map = scan.generate_maps(ox, oy)
            space_map = scan.roll_map(space_map, x-4.5, y-4.5)
            obstacle_map = scan.roll_map(obstacle_map, x-4.5, y-4.5)
        else: 
            _space_map, _obstacle_map = scan.generate_maps(ox, oy)
            _space_map = scan.roll_map(_space_map, x-4.5, y-4.5)
            _obstacle_map = scan.roll_map(_obstacle_map, x-4.5, y-4.5)

            space_map = np.minimum(_space_map, space_map)
            obstacle_map = np.maximum(_obstacle_map, obstacle_map)

        occupancy_map = scan.merge_maps(space_map, obstacle_map)

        plt.clf()
        plt.imshow(occupancy_map, origin='lower', cmap="PiYG_r")
        plt.plot(int(round((x-4.5 - scan.min_w) / scan.resolution)), int(round((y-4.5 - scan.min_w) / scan.resolution)), "ob")
        plt.pause(0.1)

    np.save('gridmap_maze3', occupancy_map)

def loadMapCsv():
    file_name = 'gridmap_maze3.npy'
    occupancy_map = np.load(file_name)
    plt.imshow(occupancy_map, origin='lower', cmap="PiYG_r")

    plt.show()

def calcCompleteMap():
    file_name = 'gridmap_maze3.npy'
    world_map = np.load(file_name)
    
    buffer_size=10
    thr_distance=1.0
    noveltyBuffer = NoveltyBuffer(buffer_size, thr_distance)

    # print(np.count_nonzero(world_map == 0))
    # print(np.count_nonzero(world_map == 1))
    # print(np.count_nonzero(world_map == 0.5))

    env = maze3Env()
    env.setting()

    # fig = plt.figure()
    # ax = plt.axes()
    fig, ax = plt.subplots()
    x, y = 0, 0
    occupancy_map = None

    completeRate = 0
    old_occupancy_map_space_num = 0

    done = False

    while completeRate < 0.99 and not done:
        # state = env.reset()
        action = env.sample_random_action()
        action[0] = 1.0 
        state, reward, done, _ = env.step(action)
        # state, reward, done, _ = env.step([1,1,0])
        x, y = state[0], state[1]

        scan = GridScan(lidar_maxLen=10.0, resolution=0.1)

        rads, dist = env.scan()
        ox = np.sin(rads) * dist
        oy = np.cos(rads) * dist

        if occupancy_map is None:
            space_map, obstacle_map = scan.generate_maps(ox, oy)
            space_map = scan.roll_map(space_map, x-4.5, y-4.5)
            obstacle_map = scan.roll_map(obstacle_map, x-4.5, y-4.5)
        else: 
            _space_map, _obstacle_map = scan.generate_maps(ox, oy)
            _space_map = scan.roll_map(_space_map, x-4.5, y-4.5)
            _obstacle_map = scan.roll_map(_obstacle_map, x-4.5, y-4.5)

            space_map = np.minimum(_space_map, space_map)
            obstacle_map = np.maximum(_obstacle_map, obstacle_map)

        occupancy_map = scan.merge_maps(space_map, obstacle_map)

        # plt.clf()
        ax.clear()

        # ax.images.clear()
        plt.imshow(occupancy_map, origin='lower', cmap="PiYG_r")
        
        # ax.collections.clear()
        plt.plot(int(round((x-4.5 - scan.min_w) / scan.resolution)), int(round((y-4.5 - scan.min_w) / scan.resolution)), "ob")


        noveltyBuffer.add_if_far(state[:2])

        # ax.patches.clear()
        for p in noveltyBuffer.buffer:
            draw_circle = patches.Circle(xy=(int(round((p[0]-4.5 - scan.min_w) / scan.resolution)), int(round((p[1]-4.5 - scan.min_w) / scan.resolution))), radius=thr_distance/scan.resolution, color='r', fill=False)
            ax.add_patch(draw_circle)

        occupancy_map_space_num = np.count_nonzero((occupancy_map[occupancy_map == world_map]) == 0)
        world_map_space_num = np.count_nonzero(world_map == 0)
        completeRate = occupancy_map_space_num / world_map_space_num
        # print(f'{completeRate}:{occupancy_map_space_num}/{world_map_space_num}')
        # print(f'{completeRate}:{occupancy_map_space_num}/{world_map_space_num} -> {noveltyBuffer.calc_mean_distance(state[:2])}')
        print(f'{completeRate}:increase {(occupancy_map_space_num - old_occupancy_map_space_num)*(scan.resolution*scan.resolution)}')
        old_occupancy_map_space_num = occupancy_map_space_num

        plt.pause(0.1)
        # plt.show()

def local_map():

    # env = maze3Env()
    # env = CuriosityEnv1()
    # env = CuriosityEnv2()
    # env = CuriosityEnv3()
    env = CuriosityEnv4()
    env.setting()

    # fig = plt.figure()
    # ax = plt.axes()
    fig, ax = plt.subplots()
    x, y = 0, 0
    occupancy_map = None

    completeRate = 0
    old_occupancy_map_space_num = 0

    done = False

    # scan = GridScan(lidar_maxLen=10.0, resolution=0.1)
    # scan = GridScan(lidar_maxLen=10.0, resolution=0.15625)
    # scan = GridScan(lidar_maxLen=10.0, resolution=0.3125)
    scan = GridScan(lidar_maxLen=10.0, resolution=0.01953125)
    # scan = GridScan(lidar_maxLen=10.0, resolution=0.009765625)
    
    print(scan.grid_num)

    for _ in range(5):

        while not done:
            # state = env.reset()
            action = env.sample_random_action()
            # action[0] = 1.0 
            state, reward, done, _ = env.step(action)
            # state, reward, done, _ = env.step([1,1,0])
            x, y = state[0], state[1]

            rads, dist = env.scan()
            ox = np.sin(rads) * dist
            oy = np.cos(rads) * dist

            if occupancy_map is None:
                space_map, obstacle_map = scan.generate_maps(ox, oy)
                _space_map, _obstacle_map = space_map, obstacle_map
                # space_map = scan.roll_map(space_map, x-4.5, y-4.5)
                # obstacle_map = scan.roll_map(obstacle_map, x-4.5, y-4.5)
            else: 
                _space_map, _obstacle_map = scan.generate_maps(ox, oy)
                # _space_map = scan.roll_map(_space_map, x-4.5, y-4.5)
                # _obstacle_map = scan.roll_map(_obstacle_map, x-4.5, y-4.5)

                space_map = np.minimum(_space_map, space_map)
                obstacle_map = np.maximum(_obstacle_map, obstacle_map)

            occupancy_map = scan.merge_maps(space_map, obstacle_map)
            _occupancy_map = scan.merge_maps(_space_map, _obstacle_map)

            # plt.clf()
            ax.clear()

            plt.imshow(_occupancy_map, cmap = "gray")

            plt.pause(0.1)
            # plt.show()

        env.reset()
        done = False

if __name__ == '__main__':
    # saveMapCsv() 
    # loadMapCsv()
    # calcCompleteMap()
    local_map()
