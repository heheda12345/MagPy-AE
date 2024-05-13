from utils import *
import itertools

LOG_DIR = '../logs/figure15'

model_names = ['bert', 'deberta', 'monodepth', 'quantized', 'tridentnet']
sys_names = ['eager', ('dynamo', 'sys'), ('xla', 'sys-xla')]
legend_orders = [0, 2, 4, 1, 3]


def parse_data(batch_size):
  data = {}
  for sys_name in itertools.chain([sys_names[0]], *sys_names[1:]):
    print("parsing", sys_name)
    data[sys_name] = []
    for model in model_names:
      t = parse_time_s(f'{LOG_DIR}/{model}.{batch_size}.{sys_name}.log')
      if t is None:
        print(f"Failed to parse {model}.{batch_size}.{sys_name}.log")
        t = np.nan
      data[sys_name].append(t)
  return data

def plot(data, model_names, sys_names, legend_orders):
  figsize = {
      "figure.figsize": (6, 3),
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

  width = len(sys_names) + 1
  x = np.array(range(len(model_names)))

  pair_color_def = [
    # (dark, light)
    ('#f4b183', None),
    ('#54b345', '#c5e0b4'),
    ('#624c7c', '#bebada'),
  ]

  pair_hatch_def = [
    ('', None),
    ('//', '//'),
    ('\\\\', '\\\\'),
  ]

  for i in range(len(sys_names)):
    sys_name = sys_names[x[i]]
    if isinstance(sys_name, tuple):
      notfull_sys_name, full_sys_name = sys_name
      notfull_perf = data[notfull_sys_name]#[model_names]
      full_perf = data[full_sys_name]#[model_names]

      plt.bar(x * width + i, notfull_perf, label=SYS_NAME[notfull_sys_name], color=pair_color_def[i][1], hatch=pair_hatch_def[i][0], width=1, edgecolor='k')
      plt.bar(x * width + i, full_perf, label=SYS_NAME[full_sys_name], color=pair_color_def[i][0], hatch=pair_hatch_def[i][1], width=1, edgecolor='k')
    else:
      perf = data[sys_name]#[model_names]
      plt.bar(x * width + i, perf, label=SYS_NAME[sys_name], color=pair_color_def[i][0], hatch=pair_hatch_def[i][0], width=1, edgecolor='k')
  
  ax.set_xticks(x * width + 1, [MODEL_NAME[n] for n in model_names])
  ax.set_ylabel("Time (s)")
  ax.set_ylim(0, 0.036)

  handles, labels = plt.gca().get_legend_handles_labels()
  plt.legend([handles[idx] for idx in legend_orders],[labels[idx] for idx in legend_orders], loc='upper left', ncol=2, frameon=False) 
  fig.savefig(PLOT_DIR + "figure1.pdf", bbox_inches='tight')




SYS_NAME = {k: v.replace(OUR_SYS, 'Fullgraph') for k, v in SYS_NAME.items()}
data = parse_data(1)
print(data)
plot(data, model_names, sys_names, legend_orders)