import torch
import numpy as np


def gen_sliding_lights(samples=120):
    lights = []
    for i in range(samples):
        phi = np.pi / 2
        theta = - np.pi / 2 + np.pi / (samples-1) * i
        y = np.cos(phi)
        x = np.sin(phi) * np.sin(theta)
        z = np.sin(phi) * np.cos(theta) - 1
        lights += [torch.from_numpy(np.array([[x, y, z]])).float().cuda()]
    return lights


def gen_circular_lights(theta=np.pi/6, samples=120):
    z = np.sin(theta) - 1
    r = np.cos(theta)
    lights = []
    for i in range(samples):
        phi = np.pi * 2 / (samples-1) * i
        x = np.sin(phi) * r
        y = np.cos(phi) * r
        lights += [torch.from_numpy(np.array([[x, y, z]])).float().cuda()]
    return lights