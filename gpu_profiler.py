import re
import os
import csv
import io
from io import StringIO
import sys
import argparse
import subprocess


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # 跟run.py保持一致
  parser.add_argument("--compile", type=str, default="sys")
  parser.add_argument("--model", type=str, required=True)
  parser.add_argument("--bs", type=int, default=1)
  parser.add_argument("--dyn_cf", action="store_true")
  parser.add_argument("--dyn_bs", action="store_true")
  parser.add_argument("--dyn_len", action="store_true")
  parser.add_argument("--repeat", type=int, default=100)
  parser.add_argument("--no_check", dest="check", action="store_false")
  args = parser.parse_args()
  args_to_pass = sys.argv[1:]

  nsys_output = f'{args.model}.{args.bs}.{args.compile}'
  # ！需要清理已有的
  def check_and_clean(file):
    if os.path.exists(file):
      os.remove(file)
      print(f"previous {file} existed, removed", flush=True)
  check_and_clean(f"{nsys_output}.nsys-rep")
  check_and_clean(f"{nsys_output}.sqlite")

  profiler_cmd = ['nsys', 'profile', f'--output={nsys_output}', '--capture-range=cudaProfilerApi', '--force-overwrite=true']
  run_cmd = profiler_cmd + ['python', 'run.py'] + args_to_pass
  print(f"{run_cmd=}")
  profile_proc = subprocess.run(run_cmd, capture_output=True, text=True)
  print(f"{profile_proc.returncode=}")
  # print(f"{profile_proc.stdout=}")
  # print(f"{profile_proc.stderr=}")

  if profile_proc.returncode != 0 and profile_proc.returncode != 143:
    print(profile_proc.stdout)
    print(profile_proc.stderr, file=sys.stderr)
    print("ERROR!!!")
    exit(-1)
  
  regex_pattern = r'(\d+)\s+iters,.*avg\s+=\s+(\d+\.\d+)\s+s'
  match = re.search(regex_pattern, profile_proc.stdout)
  assert match
  iters = int(match.group(1))
  avg_s = float(match.group(2))
  print("\n")
  print(f"compile mode: {args.compile}")
  print(f"\033[31miters: {iters}, avg: {avg_s} s\033[m")
  assert iters == args.repeat
  
  
  stats_proc = subprocess.run(['nsys', 'stats', '-q', '--report=gpukernsum', '--format=csv', f"{nsys_output}.nsys-rep"], capture_output=True, text=True)
  assert stats_proc.returncode == 0
  reader = csv.reader(io.StringIO(stats_proc.stdout.strip()), delimiter=',')
  kernel_total_time_ns = 0.0
  for line_id, row in enumerate(reader):
    # print("\t".join(row))
    if line_id == 0:
      # header
      assert row[1] == 'Total Time (ns)'
    else:
      kernel_total_time_ns += float(row[1])
  kernel_time_s = kernel_total_time_ns / iters * 1e-9
  non_kernel_time_s = avg_s - kernel_time_s
  total_time_s = avg_s
  kernel_percent = kernel_time_s / total_time_s * 100
  print(f"\033[31mkernel: {kernel_time_s:.4f} s ({kernel_percent:.2f} %), non_kernel: {non_kernel_time_s:.4f} s, total: {avg_s:.4f} s\033[m")

    

  
