import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import math
import re

COLOR_DEF = [
  '#f4b183',
  '#c5e0b4',
  '#ffd966',
  '#bdd7ee',
  "#8dd3c7",
  "#bebada",
  "#fb8072",
]


HATCH_DEF = [
  '',
  '////',
  '\\\\\\\\',
  '..',
  'xx',
  '++',
  'oo',
]

COLOR_DEF_LINE = [
  '#f4684d',
  '#ffd93c',
  '#73e069',
  '#6fd7ee',
  '#52d375',
  '#6f6dda',
  '#fb4b43',
  '#4b68d3',
  '#fd6931',
  '#cccccc',
  '#fccde5',
  '#69de3d',
  '#ffd917',
  '#fc522c',
  '#4463cf',
  '#3c7260',
  '#f45e21',
  '#ffc91b',
  '#467447'
]

PLOT_DIR = './figures/'
# LOG_DIR = './logs/'

MODEL_NAME = {
  'align': 'ALIGN',
  'bert': 'Bert',
  'deberta': 'DeBERTa',
  'densenet': 'DenseNet',
  'monodepth': 'MonoDepth',
  'quantized': 'Quantized',
  'resnet': 'ResNet',
  'tridentnet': 'TridentNet',
}

OUR_SYS = 'MagPy'

SYS_NAME = {
  'eager': 'Eager',
  'dynamo': 'TorchDynamo-Inductor',
  'sys': f'{OUR_SYS}-Inductor',
  'script': 'TorchScript-TorchScript',
  'sys-torchscript': f'{OUR_SYS}-TorchScript',
  'xla': 'LazyTensor-XLA',
  'dynamo-xla': 'TorchDynamo-XLA',
  'sys-xla': f'{OUR_SYS}-XLA',
}


def parse_csv(path, sep='\t'):
  data = pd.read_csv(path, sep=sep)
  data = data.set_index(data.iloc[:, 0])
  data = data.drop(columns=data.columns[0])
  data.index.name = ''
  data.columns.name = ''
  return data

def parse_time_ms(f_path):
    pattern = re.compile(r"min = (\d+\.\d+) ms, max = (\d+\.\d+) ms, avg = (\d+\.\d+) ms")
    with open(f_path) as f:
        for line in f:
            if "min" in line and "max" in line and "avg" in line:
                g = pattern.search(line)
                if g is None: return None
                t = float(g.group(3))
                return t
    return None

def parse_time_s(f_path):
    pattern = re.compile(r"min = (\d+\.\d+) s, max = (\d+\.\d+) s, avg = (\d+\.\d+) s")
    with open(f_path) as f:
        for line in f:
            if "min" in line and "max" in line and "avg" in line:
                g = pattern.search(line)
                if g is None: return None
                t = float(g.group(3))
                return t
    return None