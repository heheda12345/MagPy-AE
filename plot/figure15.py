from utils import *

LOG_DIR = '../logs/e2e'

model_names = ['align', 'bert', 'deberta', 'densenet', 'monodepth' ,'quantized', 'resnet', 'tridentnet']
sys_names = ['eager', ('dynamo', 'sys'), ('script', 'sys-torchscript'), ('xla', 'dynamo-xla', 'sys-xla')]


def parse_data(batch_size):
  data = {}
  for sys_name in SYS_NAME.keys():
    data[sys_name] = []
    for model in model_names:
      t = parse_time_s(f'{LOG_DIR}/{model}.{batch_size}.{sys_name}.log')
      if t is None:
        print(f"Failed to parse {model}.{batch_size}.{sys_name}.log")
        t = np.nan
      data[sys_name].append(t)
  return data
      


data_bs1 = parse_data(1)
data_bs16 = parse_data(16)
print("BS=1")
print(data_bs1)
print("BS=16")
print(data_bs16)

def plot(data, model_names, sys_names, figure_name):
  figsize = {
      "figure.figsize": (12, 2),
      # 'font.sans-serif': 'Times New Roman',
      'axes.labelsize': 12,
      'font.size': 12,
      'legend.fontsize': 10,
      'xtick.labelsize': 12,
      'ytick.labelsize': 12,
      'pdf.fonttype': 42,
      'ps.fonttype': 42
  }
  plt.rcParams.update(figsize)
  fig, ax = plt.subplots()


  pair_color_def = [
    # (dark -> light)
    ('#f4b183',),
    ('#54b345', '#c5e0b4'),
    ('#624c7c', '#bebada'),
    ('#2878b5', '#70aedb', '#bdd7ee'),
  ]

  max_sys_num_for_same_backend = max(len(sys) if isinstance(sys, tuple) else 1 for sys in sys_names)
  print(f"{max_sys_num_for_same_backend=}")

  total_sys_num = sum(len(sys) if isinstance(sys, tuple) else 1 for sys in sys_names)
  width = total_sys_num + 2 
  x = np.array(range(len(model_names)))

  sys_idx = 0
  off = 0.6
  def generate_empty_legend(num):
    for _ in range(num):
      plt.bar(np.NaN, np.NaN, color='w', label=' ')

  for i in range(len(sys_names)):
    sys_name = sys_names[x[i]]
    if isinstance(sys_name, tuple):
      for sub_idx, sub_sys_name in enumerate(sys_name):
        perf = data[sub_sys_name]#[model_names]
        y = list(perf)
        idx = x * width + sys_idx - off
        plt.bar(idx, y, label=SYS_NAME[sub_sys_name], color=pair_color_def[i][sub_idx], hatch='////' if OUR_SYS in SYS_NAME[sub_sys_name] else '', width=1, edgecolor='k')
        if np.isnan(y).any():
          for nan_idx in np.argwhere(np.isnan(y)):
            plt.text(idx[nan_idx], 0, 'Graph Compile Fail', fontsize=6, rotation=90, ha='center', va='bottom')
        sys_idx += 1
      generate_empty_legend(max_sys_num_for_same_backend - len(sys_name))
      off -= 0.3
    else:
      perf = data[sys_name]#[model_names]
      plt.bar(x * width + sys_idx - off, perf, label=SYS_NAME[sys_name], color=pair_color_def[i][0], hatch='////' if OUR_SYS in SYS_NAME[sys_name] else '', width=1, edgecolor='k')
      generate_empty_legend(max_sys_num_for_same_backend - 1)
      off -= 0.3
      sys_idx += 1
  
  ax.set_xticks(x * width + 3.5, [MODEL_NAME[n] for n in model_names])
  ax.set_ylabel("Time (s)")
  ax.set_ylim(0, 0.1)
  plt.legend() 
  plt.legend(ncol=4, frameon=False)

  fig.savefig(PLOT_DIR + f"{figure_name}.pdf", bbox_inches='tight')



plot(data_bs1, model_names, sys_names, figure_name='figure15a')
plot(data_bs16, model_names, sys_names, figure_name='figure15b')