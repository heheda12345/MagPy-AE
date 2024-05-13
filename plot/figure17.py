from utils import *

LOG_DIR = '../logs/dyn-shape'

model_names = [
  'bert-bs',
  'bert-seqlen',
  'resnet-bs',
  'deberta-bs',
  'deberta-seqlen',
  'densenet-bs',
]

sys_names = [
  # data_name, legend_name
  ('eager', 'Eager'),
  ('dynamo-dynamic', 'TorchDynamo'),
  ('sys-dynamic', OUR_SYS),
]

def parse_data():
  data = {}
  for sys_name, _ in sys_names:
    data[sys_name] = []
    for model in model_names:
      t = parse_time_s(f'{LOG_DIR}/{model}.{sys_name}.log')
      if t is None:
        print(f"Failed to parse {model}.{sys_name}.log")
        t = np.nan
      data[sys_name].append(t)
  return data

data = parse_data()
print(data)

def plot(data, model_names, sys_names):
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
  fig, ax = plt.subplots()

  width = len(sys_names) + 1
  x = np.array(range(len(model_names)))

  for i in range(len(sys_names)):
    data_name, legend_name = sys_names[i]
    y = data[data_name]#[model_names]
    plt.bar(x * width + i, y, label=legend_name, color=COLOR_DEF[i], hatch=HATCH_DEF[i], width=1, edgecolor='k')
  
  def replace_model_name(n):
    out = n
    for k, v in MODEL_NAME.items():
      if k in n:
        out = n.replace(k, v)
    return out

  ax.set_xticks(x * width + 1, [replace_model_name(n) for n in model_names], rotation=20)
  ax.set_ylabel("Time (s)")
  # plt.legend(ncol=len(sys_names))
  plt.legend(loc='center left', bbox_to_anchor=(0.2, 0.7), frameon=False)
  fig.savefig(PLOT_DIR + "figure17.pdf", bbox_inches='tight')


plot(data, model_names, sys_names)

