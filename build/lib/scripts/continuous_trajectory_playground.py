
import torch
import numpy as np

import os
import PIL.Image
import torchvision
import argparse
import json
import taichi as ti
import sym
import matplotlib.pyplot as plt
import torchvision
import pypose as pp
from pytorch_msssim import ssim
import pylab as pl
import pickle

import plotCoordinateFrame

pose_1 = sym.Pose3.from_storage([0,0,0,1,0,0,0])
pose_1 = sym.Pose3.from_storage([0,0,0,1,0,0,0])
pose_1 = sym.Pose3.from_storage([0,0,0,1,0,0,0])
pose_1 = sym.Pose3.from_storage([0,0,0,1,0,0,0])
