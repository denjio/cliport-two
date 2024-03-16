"""Data collection script."""

import os
import hydra
import numpy as np
import random

from cliport import tasks
from cliport.dataset1 import RavensDataset
from cliport.environments.environment import Environment


@hydra.main(config_path='./cfg', config_name='data')
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg['assets_root'],
        disp=cfg['disp'],
        shared_memory=cfg['shared_memory'],
        hz=480,
        record_cfg=cfg['record']
    )
    task = tasks.names[cfg['task']]()

    task.mode = cfg['mode']
    record = cfg['record']['save_video']
    save_data = cfg['save_data']

    # Initialize scripted oracle agent and dataset.
    agent = task.oracle(env)
    data_path = os.path.join(cfg['data_dir'], "{}-2complete-{}".format(cfg['task'], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    if seed < 0:
        if task.mode == 'train':
            seed = -2
        elif task.mode == 'val': # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == 'test':
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < cfg['n']:
        episode, total_reward = [], 0
        seed += 2
        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)
        print('Oracle demo: {}/{} | Seed: {}'.format(dataset.n_episodes + 1, cfg['n'], seed))
        env.set_task(task)
        obs = env.reset()
        info = env.info
        reward = 0

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == 'val' and seed > (-1 + 10000):
            raise Exception("!!! Seeds for val set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f'{dataset.n_episodes+1:06d}')

        # Rollout expert policy
        fg = 0
        for _ in range(task.max_steps):
            # print('max', task.max_steps)
            act1 = agent.act(obs, info)
            obs2, reward, done, info2 = env.step(act1)
            total_reward += reward
            origin_obs = obs
            origin_info = info
            if done:
                fg = 1
                break

            act2 = agent.act(obs2, info2)
            obs, reward1, done, info = env.step(act2)
            total_reward += reward1

            episode.append((origin_obs, act1, act2, reward+reward1, origin_info))

            lang_goal = info['lang_goal']
            print(f'Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}')
            if done:
                fg = 2
                break
        if fg == 1:
            flag_act = {'pose0': (np.array([0, 0, 0]), np.array([0, 0, 0, 1])),
                        'pose1': (np.array([0, 0, 0]), np.array([0, 0, 0, 1]))}
            episode.append((obs, act1, flag_act, reward, origin_info))
            episode.append((obs, None, None, reward + reward1, origin_info))
        if fg == 2:
            episode.append((obs, None, None, reward+reward1, origin_info))
        # print('obs', len(obs['color']), obs['color'][0].shape, len(obs['depth']), obs['depth'][0].shape)
        # print('act',act)
        # print('reward', reward)
        # print('info', info.keys(),info)

        # print(episode)
        # End video recording
        if record:
            env.end_rec()

        # Only save completed demonstrations.
        if save_data and total_reward > 0.99:
            dataset.add(seed, episode)


if __name__ == '__main__':
    main()
