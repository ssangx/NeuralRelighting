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


def gen_uniform_in_hemisphere():
    """
    Randomly generate an unit 3D vector to represent
    the light direction
    """
    phi = np.random.uniform(0, np.pi * 2)
    costheta = np.random.uniform(0, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta) - 1
    return torch.from_numpy(np.array([x, y, z])).float().cuda()