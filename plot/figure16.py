from utils import *

LOG_DIR = '../logs/profile'

model_names = ['align', 'bert', 'deberta', 'densenet', 'monodepth' ,'quantized', 'resnet', 'tridentnet']

def parse_data():
  data = {
    'graph compilation (inductor)': [],
    'sys analyzation': [],
    'graph match': [],
    'graph execution': [],
  }
  for model in model_names:
    d = {}
    with open(LOG_DIR + f"/{model}.log") as f:
      for s in f.readlines():
        if "compile time:" in s:
          t = float(s.split()[-2])
          d['graph compilation (inductor)'] = t
        elif "first run" in s:
          t = float(s.split()[-2])
          d['sys analyzation'] = t - d['graph compilation (inductor)']
        elif "e2e" in s:
          t = float(s.split()[-2])
          d['graph match'] = t
        elif "graph profile" in s:
          t = float(s.split()[-2])
          d['graph execution'] = t
          d['graph match'] = d['graph match'] - t
    if len(d) != 4:
      print("parse fail:", model)
      raise ValueError
    else:
      for k in d:
        data[k].append(d[k])
  return data



def plot(data, model_names, case_names, time_unit, figure_name):
  figsize = {
      "figure.figsize": (6, 1),
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

  width = len(model_names) + 1
  x = np.array(range(len(model_names)))


  y_off = np.zeros(len(model_names), dtype=float)
  for i in range(len(case_names)):
    data_name, legend_name = case_names[i]
    y = data[data_name]#[model_names]
    plt.bar(x * width, y, bottom=y_off, label=legend_name, color=COLOR_DEF[i], hatch=HATCH_DEF[i], width=4, edgecolor='k')
    y_off += y
  

  ax.set_xticks(x * width, [MODEL_NAME[n] for n in model_names], rotation=20)
  ax.set_ylabel(f"Time ({time_unit})")
  legend_orders = list(range(len(case_names)))
  legend_orders.reverse()
  handles, labels = plt.gca().get_legend_handles_labels()
  plt.legend([handles[idx] for idx in legend_orders],[labels[idx] for idx in legend_orders], ncol=len(case_names), frameon=False) 
  fig.savefig(PLOT_DIR + f"{figure_name}.pdf", bbox_inches='tight')

data = parse_data()
for k, v in data.items():
  print(k, v)
compile_case_names = [
  # (data_name, legend_name)
  ('graph compilation (inductor)', 'graph compilation (Inductor)'),
  ('sys analyzation', f'{OUR_SYS} profile'),
]
run_case_names = [
  # (data_name, legend_name)
  ('graph execution', 'mock execution'),
  ('graph match', 'guard validation'),
]

plot(data, model_names, compile_case_names, time_unit='s', figure_name='figure16a')
plot(data, model_names, run_case_names, time_unit='ms', figure_name='figure16b')