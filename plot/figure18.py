from utils import *

LOG_DIR = '../logs/control_flow'

model_names = [
  # (data_name, legend_name)
  ('lstm', 'LSTM'),
  ('blockdrop', 'BlockDrop'),
]

sys_names = [
  # (data_name, legend_name)
  ('eager', 'Eager'),
  ('dynamo-nnf', 'TorchDynamo'),
  ('sys-nnf', OUR_SYS),
]

def parse_data(batch_size):
  data = {}
  for sys_name in [s[0] for s in sys_names]:
    data[sys_name] = []
    for model in [n[0] for n in model_names]:
      t = parse_time_ms(f'{LOG_DIR}/{model}.{batch_size}.{sys_name}.log')
      if t is None:
        t = parse_time_s(f'{LOG_DIR}/{model}.{batch_size}.{sys_name}.log')
        if t is None:
          print(f"Failed to parse {model}.{batch_size}.{sys_name}.log")
          t = np.nan
        else:
          t *= 1000
      data[sys_name].append(t)
  return data

data_bs1 = parse_data(1)
data_bs16 = parse_data(16)

print("BS=1")
print(data_bs1)
print("BS=16")
print(data_bs16)

def _plot(data, model_names, sys_names, ax1, ax2):
  width = len(sys_names) + 1
  x = np.array(range(len(model_names)))
  max_y = -1
  for i in range(len(sys_names)):
    data_name, legend_name = sys_names[i]
    y = data[data_name]#[[n[0] for n in model_names]]
    max_y = max(max(y), max_y)
    ax1.bar(x * width + i, y, label=legend_name, color=COLOR_DEF[i], hatch=HATCH_DEF[i], width=1, edgecolor='k')
    ax2.bar(x * width + i, y, label=legend_name, color=COLOR_DEF[i], hatch=HATCH_DEF[i], width=1, edgecolor='k')
  
  # print(f"{max_y=}")
  ax1_range = (max_y * 0.9, int(max_y * 1.1) // 2 * 2)
  # print(f"{ax1_range=}")
  ax2_range = (0, max_y * 0.3)
  ax1.set_ylim(ax1_range[0], ax1_range[1])
  ax2.set_ylim(ax2_range[0], ax2_range[1])

  for i in range(len(sys_names)):
    data_name, legend_name = sys_names[i]
    y = data[data_name]#[[n[0] for n in model_names]]
    for x_loc, y_loc in zip(x * width + i, y):
      if y_loc < ax1_range[1] and y_loc >= ax1_range[0]:
        ax1.text(x_loc, y_loc, f'{y_loc:.2f}', ha='center', va='bottom',fontsize=10)
      else:
        ax2.text(x_loc, y_loc, f'{y_loc:.2f}', ha='center', va='bottom',fontsize=10)

  ax1.spines.bottom.set_visible(False)
  ax2.spines.top.set_visible(False)
  ax1.xaxis.set_visible(False)
  ax1.tick_params(labeltop=False)
  ax2.xaxis.tick_bottom()
  ax2.set_xticks(x * width + 1, [n[1] for n in model_names])
  d = .4  # proportion of vertical to horizontal extent of the slanted line
  kwargs = dict(marker=[(-1, -d), (1, d)], markersize=6, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
  ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
  ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

def plot(data_bs1, data_bs16, model_names, sys_names):
  # ref: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
  figsize = {
      "figure.figsize": (6, 2),
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
  fig, axs = plt.subplots(2, 2)
  _plot(data_bs1, model_names, sys_names, axs[0, 0], axs[1, 0])
  _plot(data_bs16, model_names, sys_names, axs[0, 1], axs[1, 1])

  axs[1, 0].set_xlabel('BS=1')
  axs[1, 1].set_xlabel('BS=16')
  fig.supylabel('Time (ms)')
  axs[0, 1].legend(loc='upper right', frameon=False)
  fig.savefig(PLOT_DIR + f"figure18.pdf", bbox_inches='tight')


plot(data_bs1, data_bs16, model_names, sys_names)
