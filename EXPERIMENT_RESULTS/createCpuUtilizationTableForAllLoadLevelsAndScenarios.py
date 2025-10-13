# Import required libraries
import pandas as pd
import numpy as np

from pathlib import Path
import glob
from collections import defaultdict
import os
import json


def get_jmeter_time_bounds(jmeter_dir, trim_seconds):
    """
    Find the jmeter .jtl file in the given directory, read the timeStamp column,
    and return the min and max timestamps, trimmed by trim_seconds (in seconds).
    Returns (trimmed_start, trimmed_end) as pandas.Timestamp, or (None, None) if not found.
    """
    # Find the first .jtl file in the directory or subdirectories
    jtl_file = None
    for root, dirs, files in os.walk(jmeter_dir):
        for f in files:
            if f.endswith('.jtl'):
                jtl_file = os.path.join(root, f)
                break
        if jtl_file:
            break
    if not jtl_file:
        return None, None
    try:
        #print(f"Reading jmeter file: {jtl_file}")
        df = pd.read_csv(jtl_file)
        if 'timeStamp' not in df.columns:
            return None, None
        min_ts = df['timeStamp'].min()
        max_ts = df['timeStamp'].max()
        # Convert ms to pandas.Timestamp
        min_time = pd.to_datetime(min_ts, unit='ms')
        max_time = pd.to_datetime(max_ts, unit='ms')
        # Trim by trim_seconds
        trimmed_start = min_time + pd.Timedelta(seconds=trim_seconds)
        trimmed_end = max_time - pd.Timedelta(seconds=trim_seconds)
        return trimmed_start, trimmed_end
    except Exception as e:
        print(f"Error reading jmeter file {jtl_file}: {e}")
        return None, None


def trim_time_series(df, trim_seconds, jmeter_bounds=None):
    """
    Removes records outside the time window defined by jmeter_bounds (start, end).
    If jmeter_bounds is None, falls back to old behavior (relative trim).
    """
    if jmeter_bounds is not None and all(jmeter_bounds):
        start, end = jmeter_bounds
        return df[(df['datetime'] >= start) & (df['datetime'] <= end)]
    # Fallback: old behavior - required for Load: 0
    if trim_seconds <= 0:
        return df
    start_time = df['datetime'].min()
    end_time = df['datetime'].max()
    cutoff_start = start_time + pd.Timedelta(seconds=trim_seconds)
    cutoff_end = end_time - pd.Timedelta(seconds=trim_seconds)
    return df[(df['datetime'] >= cutoff_start) & (df['datetime'] <= cutoff_end)]



def extract_service_pids(experiment_log_path):
    """
    Reads the experiment_log.jsonl file and returns the list of service_pids (as strings).
    Returns an empty list if not found.
    """
    try:
        with open(experiment_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'service_pids' in entry:
                        return [str(pid) for pid in entry['service_pids']]
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading {experiment_log_path}: {e}")
    return []


def parse_procfs_data(procfs_file, service_pids, n_cores=80, ticks_per_sec=100, jmeter_bounds=None):
    """
    Parses procfs CSV and returns:
      - proc_util: DataFrame with ['datetime', 'util_ratio'] (summed across all service_pids)
      - sys_df: DataFrame with system CPU utilization info
      - mem_deltas: DataFrame with ['datetime', 'pid', 'delta_vmsize', 'delta_vmrss']
      - io_deltas: DataFrame with ['datetime', 'pid', 'delta_read_bytes', 'delta_write_bytes']
    """
    import pandas as pd
    try:
        df = pd.read_csv(procfs_file)
        # Determine steady-state window (jmeter_bounds) or fallback to trim 60s from start/end
        min_time = pd.to_datetime(df['Timestamp'].min(), unit='ms')
        max_time = pd.to_datetime(df['Timestamp'].max(), unit='ms')
        trim_seconds = 60
        if jmeter_bounds is not None and all(jmeter_bounds):
            start, end = jmeter_bounds
        else:
            start = min_time + pd.Timedelta(seconds=trim_seconds)
            end = max_time - pd.Timedelta(seconds=trim_seconds)
        # Filter df to steady-state window
        df = df[(pd.to_datetime(df['Timestamp'], unit='ms') >= start) & (pd.to_datetime(df['Timestamp'], unit='ms') <= end)]
        # --- CPU (stat) ---
        stat_mask = df['SourceFile'].str.endswith(r'stat')
        stat_df = df[stat_mask].copy()
        stat_df['pid'] = stat_df['SourceFile'].str.extract(r'/proc/(\d+)/stat')
        stat_df['datetime'] = pd.to_datetime(stat_df['Timestamp'], unit='ms')
        stat_df['userTime_s'] = stat_df['userTime (Ticks)'] / ticks_per_sec
        stat_df['systemTime_s'] = stat_df['systemTime (Ticks)'] / ticks_per_sec
        sys_df = stat_df[stat_df['SourceFile'] == '/proc/stat'].sort_values('datetime')
        proc_data = stat_df[stat_df['SourceFile'] != '/proc/stat']
        proc_df = proc_data[proc_data['pid'].isin(service_pids)].sort_values(['pid', 'datetime'])
        sys_df = sys_df[['datetime', 'userTime_s', 'systemTime_s']].copy()
        sys_df['total_cpu'] = sys_df['userTime_s'] + sys_df['systemTime_s']
        sys_df['delta_cpu'] = sys_df['total_cpu'].diff()
        sys_df['interval'] = sys_df['datetime'].diff().dt.total_seconds()
        sys_df = sys_df.iloc[1:]
        return None, sys_df, None, None
    except Exception as e:
        print(f"Error parsing procfs file {procfs_file}: {e}")
        return None, None, None, None


def main():
    """
    Main entry point: collects data, generates and saves the boxplot PDF for all load levels.
    """
    trim_seconds = 60  # Number of seconds to trim at start and end of each time series
    # List of scenario suffixes to include and order
    scenario_suffixes = [
        "docker_tools",
        #"idle_no_tools",
        #"docker_none",
        #"docker_idle",
        "docker_kepler",
        "docker_scaphandre",
        "docker_otjae",
        "docker_joularjx"
    ]
    # Optional: custom labels for scenario suffixes
    custom_labels = {
        "idle_no_tools": "Idle",
        "docker_tools": "",
        "docker_none": "Container\nIdle (CI)",
        "docker_idle": "CI, Powercap (PC)\nand ProcFS",
        "docker_kepler": "CI, PC, ProcFS\nand Kepler",
        "docker_scaphandre": "CI, PC, ProcFS\nand Scaphandre",
        "docker_otjae": "CI, PC, ProcFS\nand OTJAE",
        "docker_joularjx": "CI, PC, ProcFS\nand JoularJX"
    }


    # Minimal replacement for iterating load levels and scenarios
    exp_results = Path("./EXPERIMENT_RESULTS")
    all_dirs = [d for d in exp_results.iterdir() if d.is_dir()]
    # Group by numeric prefix (load level)
    load_level_map = defaultdict(list)
    for d in all_dirs:
        if '_' in d.name:
            prefix = d.name.split('_')[0]
        else:
            prefix = d.name
        if prefix:
            load_level_map[prefix].append(d)
    data_by_load = defaultdict(dict)
    for load_level, dirs in load_level_map.items():
        for run_path in dirs:
            for scenario_dir in run_path.iterdir():
                if not scenario_dir.is_dir():
                    continue
                scenario_name = scenario_dir.name
                if scenario_suffixes is not None and not any(scenario_name.endswith(suf) for suf in scenario_suffixes):
                    continue
                data_by_load[load_level][scenario_name] = {}

    # --- LaTeX Table Generation ---
    # Only keep numeric load level keys (exclude e.g. 'pcpumin', 'pcpumax')
    numeric_items = [(k, v) for k, v in data_by_load.items() if k.isdigit()]
    sorted_loads = sorted(numeric_items, key=lambda x: int(x[0]))
    # Define scenario suffixes and LaTeX column order
    scenario_columns = [
        ("docker_tools", "CPU\\textsubscript{none}"),
        ("docker_kepler", "CPU\\textsubscript{Kepler}"),
        ("docker_scaphandre", "CPU\\textsubscript{Scaphandre}"),
        ("docker_otjae", "CPU\\textsubscript{OTJAE}"),
        ("docker_joularjx", "CPU\\textsubscript{JoularJX}")
    ]
    print("\\begin{table*}")
    print("    \\begin{center}")
    print("        \\begin{tabular}{ |c|c|c|c|c|c| }")
    print("            \\hline")
    print("            Load & " + " & ".join([col for _, col in scenario_columns]) + "\\\\")
    print("            \\hline")
    for load_level, scenario_dict in sorted_loads:
        row = []
        for suffix, _ in scenario_columns:
            cpu_utils = []
            exp_results = Path("./EXPERIMENT_RESULTS")
            for d in exp_results.iterdir():
                if d.is_dir() and d.name.startswith(str(load_level)):
                    for sub in d.iterdir():
                        if sub.is_dir() and sub.name.endswith(suffix):
                            # Try to get jmeter bounds for steady state
                            jmeter_bounds = get_jmeter_time_bounds(str(sub), 60)
                            experiment_log_path = sub / 'logs' / 'experiment_log.jsonl'
                            service_pids = []
                            if experiment_log_path.exists():
                                service_pids = extract_service_pids(str(experiment_log_path))
                            files = list(sub.glob('**/procfs_*.csv'))
                            for procfs_file in files:
                                try:
                                    _, sys_df, _, _ = parse_procfs_data(str(procfs_file), service_pids, 80, 100, jmeter_bounds)
                                    if sys_df is not None and not sys_df.empty:
                                        sys_df['cpu_util'] = sys_df['delta_cpu'] / sys_df['interval'] / 80 * 100
                                        cpu_util_val = sys_df['cpu_util'].mean()
                                        cpu_utils.append(cpu_util_val)
                                except Exception as e:
                                    print(f"Error parsing procfs for CPU util: {e}")
            if cpu_utils:
                row.append(f"{np.mean(cpu_utils):.2f}\\%")
            else:
                row.append("-")
        load_label = f"{int(load_level)*3}T/s"
        print(f"            {load_label} & " + " & ".join(row) + " \\")
        print("            \\hline")
    print("        \\end{tabular}")
    print("        \\caption{Mean CPU utilization of measurement runs on different test setups}")
    print("        \\label{tab:overhead_evaluation}")
    print("    \\end{center}")
    print("\\end{table*}")


# Run the script if executed directly
if __name__ == "__main__":
    main()
