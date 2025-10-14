# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def load_rittal_data(file_path, trim_seconds=0, jmeter_bounds=None):
    """
    Loads Rittal CSV data, sums up power values per timestamp, and trims the time series if needed.
    """
    df = pd.read_csv(file_path)
    # Group by timestamp and sum power values
    power_data = df.groupby('Timestamp')['Power (Watts)'].sum().reset_index()
    power_data['datetime'] = pd.to_datetime(power_data['Timestamp'], unit='ms')
    # Optionally trim the time series
    if trim_seconds > 0 or jmeter_bounds is not None:
        power_data = trim_time_series(power_data, trim_seconds, jmeter_bounds)
    return power_data


def calculate_power_from_energy(df, trim_seconds=0, jmeter_bounds=None):
    """
    Calculates power from energy values in powercap data, handling counter overflows and trimming.
    """
    # Sort by timestamp and domain
    df = df.sort_values(['Timestamp', 'Domain'])
    df['datetime'] = pd.to_datetime(df['Timestamp'], unit='ms')
    power_data = []
    for domain in df['Domain'].unique():
        domain_data = df[df['Domain'] == domain].copy()
        # Calculate energy and time differences
        energy_diff = domain_data[' Energy (micro joules)'].diff()
        time_diff = domain_data['datetime'].diff().dt.total_seconds()
        # Calculate power in watts
        power = energy_diff / (time_diff * 1_000_000)
        # Handle negative power values (counter overflows)
        if (power < 0).any():
            negative_indices = power[power < 0].index
            for idx in negative_indices:
                print(f"Negative power value for Domain {domain} at {domain_data.loc[idx, 'datetime']}")
                power[idx] = np.nan
        domain_data['Power'] = power
        domain_data = domain_data.dropna(subset=['Power'])
        power_data.append(domain_data)
    # Combine all domains
    result = pd.concat(power_data)
    # Only keep timestamps present in all domains
    domain_counts = result.groupby('datetime')['Domain'].count()
    complete_timestamps = domain_counts[domain_counts == len(df['Domain'].unique())].index
    result_filtered = result[result['datetime'].isin(complete_timestamps)]
    power_sum = result_filtered.groupby('datetime')['Power'].sum().reset_index()
    # Optionally trim the time series
    if trim_seconds > 0 or jmeter_bounds is not None:
        power_sum = trim_time_series(power_sum, trim_seconds, jmeter_bounds)
    return power_sum


def load_power_data(file_path, trim_seconds=0, jmeter_bounds=None):
    """
    Loads energy data based on file type and converts to power data.
    """
    if 'rittal' in file_path.lower():
        return load_rittal_data(file_path, trim_seconds, jmeter_bounds)
    elif 'powercap' in file_path.lower():
        df = pd.read_csv(file_path)
        return calculate_power_from_energy(df, trim_seconds, jmeter_bounds)
    else:
        raise ValueError(f"Unknown file type: {file_path}")


def collect_data_by_load_level(trim_seconds=0, scenario_suffixes=None):
    """
    Collects and groups all Rittal and Powercap data by load level (numeric prefix of directory name).
    Aggregates all runs (e.g., 350, 350_run2, 350_run3) for each load level.
    Only includes scenario subdirectories matching scenario_suffixes if provided.
    Returns a dict: {load_level: {scenario: {'rittal': [series...], 'powercap': [series...]}}}
    """
    exp_results = Path("./EXPERIMENT_RESULTS")
    all_dirs = [d for d in exp_results.iterdir() if d.is_dir()]
    # Group by numeric prefix (load level)
    load_level_map = defaultdict(list)
    for d in all_dirs:
        # Extract load level as the substring before '_' if present, else use the full directory name
        if '_' in d.name:
            prefix = d.name.split('_')[0]
        else:
            prefix = d.name
        if prefix:
            load_level_map[prefix].append(d)
    # Now, for each load level, look for scenario subdirectories
    data_by_load = defaultdict(lambda: defaultdict(lambda: {'rittal': [], 'powercap': []}))
    for load_level, dirs in load_level_map.items():
        for run_path in dirs:
            # Find all subdirectories (scenarios) in this run directory
            for scenario_dir in run_path.iterdir():
                if not scenario_dir.is_dir():
                    continue
                scenario_name = scenario_dir.name
                # Filter for scenario suffixes if provided
                if scenario_suffixes is not None and not any(scenario_name.endswith(suf) for suf in scenario_suffixes):
                    continue
                # Get jmeter bounds for this scenario
                jmeter_bounds = get_jmeter_time_bounds(str(scenario_dir), trim_seconds)
                # Find all Rittal and Powercap CSV files in this scenario directory
                rittal_files = list(scenario_dir.glob('**/rittal_*.csv'))
                powercap_files = list(scenario_dir.glob('**/powercap_*.csv'))
                # Load and append Rittal data
                for file_path in rittal_files:
                    try:
                        power_data = load_rittal_data(str(file_path), trim_seconds, jmeter_bounds)
                        if 'Power (Watts)' in power_data.columns:
                            data_by_load[load_level][scenario_name]['rittal'].append(power_data['Power (Watts)'])
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                # Load and append Powercap data
                for file_path in powercap_files:
                    try:
                        power_data = load_power_data(str(file_path), trim_seconds, jmeter_bounds)
                        if 'Power' in power_data.columns:
                            data_by_load[load_level][scenario_name]['powercap'].append(power_data['Power'])
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
    return data_by_load


def plot_all_load_levels_in_one_row(data_by_load, output_path, scenario_suffixes=None, custom_labels=None):
    """
    Plots a single row of boxplots, one for each load level, each showing Rittal and Powercap data for selected scenarios.
    Groups all data for each scenario (across all runs) into a single boxplot per scenario per load level.
    scenario_suffixes: list of scenario suffixes to include and order.
    custom_labels: dict mapping scenario suffix to label.
    """
    # Sort load levels numerically
    sorted_loads = sorted(data_by_load.items(), key=lambda x: int(''.join(filter(str.isdigit, x[0]))))
    n_levels = len(sorted_loads)
    fig_width = min(18, 3.5 * n_levels)  # Increase max width and per-subplot width
    fig_height = 9  # Increase height
    fig, axes = plt.subplots(1, n_levels, figsize=(fig_width, fig_height), sharey=True)
    
    if n_levels == 1:
        axes = [axes]
    for ax, (load_level, scenario_dict) in zip(axes, sorted_loads):
        box_data = []
        box_labels = []
        # Use scenario_suffixes for order and filtering
        for suffix in (scenario_suffixes if scenario_suffixes is not None else scenario_dict.keys()):
            # Find scenario(s) matching this suffix
            matching = [k for k in scenario_dict.keys() if k.endswith(suffix)] if scenario_suffixes else [suffix]
            # Group all data for this scenario across all runs
            all_rittal = []
            all_powercap = []
            for scenario in matching:
                all_rittal.extend(scenario_dict[scenario]['rittal'])
                all_powercap.extend(scenario_dict[scenario]['powercap'])
            # Combine all Rittal data for this scenario
            if all_rittal:
                combined_rittal = pd.concat(all_rittal, ignore_index=True)
                label = custom_labels.get(suffix, suffix) if custom_labels else suffix
                box_data.append(combined_rittal)
                box_labels.append(f'{label}$P_{{EM}}$')
            # Combine all Powercap data for this scenario
            if all_powercap:
                combined_powercap = pd.concat(all_powercap, ignore_index=True)
                label = custom_labels.get(suffix, suffix) if custom_labels else suffix
                box_data.append(combined_powercap)
                box_labels.append(f'{label}$P_{{S}}$')
        # Plot boxplot if data exists
        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, medianprops=dict(color="grey", linewidth=2.5),
                            widths=0.7,
                            showmeans=True,
                            meanprops={"marker": "x", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 10})
            ax.set_xticklabels(box_labels, rotation=0, ha='right', fontsize=18)
            for patch in bp['boxes']:
                patch.set_facecolor('lightgrey')
            ax.set_title(f'Load: {int(load_level) * 3} T/s', fontsize=22)
            ax.set_xlabel('', fontsize=18)
            ax.tick_params(axis='y', labelsize=16)
            ax.tick_params(axis='x', labelsize=16)
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    # Set y-label for the first subplot
    axes[0].set_ylabel('Power (Watts)', fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18, top=0.90, left=0.07, right=0.98)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved combined boxplot to {output_path}")


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
        #"docker_kepler",
        #"docker_scaphandre",
        #"docker_otjae",
        #"docker_joularjx"
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
    data_by_load = collect_data_by_load_level(trim_seconds=trim_seconds, scenario_suffixes=scenario_suffixes)
    plot_all_load_levels_in_one_row(data_by_load, "./power_consumption_boxplots_all_loads.pdf", scenario_suffixes=scenario_suffixes, custom_labels=custom_labels)

    # --- LaTeX Table Generation ---
    # Only keep numeric load level keys (exclude e.g. 'pcpumin', 'pcpumax')
    numeric_items = [(k, v) for k, v in data_by_load.items() if k.isdigit()]
    sorted_loads = sorted(numeric_items, key=lambda x: int(x[0]))
    print("\\begin{table} [h]")
    print("    \\begin{center}")
    print("        \\begin{tabular}{ |c|c|c|c|c| }")
    print("            \\hline")
    print("            Load   &  CPU\\textsubscript{UTIL} & P\\textsubscript{EM}& P\\textsubscript{S}   & P\\textsubscript{$\\Delta$}  \\")
    print("            \\hline")
    for load_level, scenario_dict in sorted_loads:
        # Get mean for each scenario
        def get_mean_power(scenario_key, subkey):
            vals = []
            for k in scenario_dict:
                if k.endswith(scenario_key):
                    vals.extend(scenario_dict[k][subkey])
            if vals:
                combined = pd.concat(vals, ignore_index=True)
                return float(combined.mean())
            return None
        # Rittal (Pem)
        Pem = get_mean_power('docker_tools', 'rittal')
        # Powercap (Ps)
        Ps = get_mean_power('docker_tools', 'powercap')
        # Delta
        Pdelta = Pem - Ps if Pem is not None and Ps is not None else None
        # Aggregate CPU utilization across all subdirectories for this load level
        cpu_utils = []
        exp_results = Path("./EXPERIMENT_RESULTS")
        for d in exp_results.iterdir():
            if d.is_dir() and d.name.startswith(str(load_level)):
                for sub in d.iterdir():
                    if sub.is_dir() and sub.name.endswith('docker_tools'):
                        experiment_log_path = sub / 'logs' / 'experiment_log.jsonl'
                        service_pids = []
                        if experiment_log_path.exists():
                            service_pids = extract_service_pids(str(experiment_log_path))
                        files = list(sub.glob('**/procfs_*.csv'))
                        for procfs_file in files:
                            try:
                                _, sys_df, _, _ = parse_procfs_data(str(procfs_file), service_pids, 80, 100, None)
                                if sys_df is not None and not sys_df.empty:
                                    sys_df['cpu_util'] = sys_df['delta_cpu'] / sys_df['interval'] / 80 * 100
                                    cpu_utils.extend(sys_df['cpu_util'].tolist())
                            except Exception as e:
                                print(f"Error parsing procfs for CPU util: {e}")
        # Format values
        def fmt(val, percent=False):
            if val is None:
                return "-"
            if percent:
                return f"{round(val, 2):.2f}\\%"
            return f"{round(val, 2):.2f}W"
        load_label = f"{int(load_level)*3}T/s"
        cpu_util_mean = np.mean(cpu_utils) if cpu_utils else None
        print(f"            {load_label} & {fmt(cpu_util_mean, percent=True)} & {fmt(Pem)} & {fmt(Ps)} & {fmt(Pdelta)} \\")
        print("            \\hline")
    print("        \\end{tabular}")
    print("        \\caption{Mean system power consumption by load level}")
    print("        \\label{tab:power_depending_on_utilization}")
    print("    \\end{center}")
    print("\\end{table}")

        # --- Delta Line Plot for Mean Pem and Ps ---
    # Collect mean Pem and Ps for each load level
    load_labels = []
    pem_means = []
    ps_means = []
    for load_level, scenario_dict in sorted_loads:
        Pem = get_mean_power('docker_tools', 'rittal')
        Ps = get_mean_power('docker_tools', 'powercap')
        if Pem is not None and Ps is not None:
            load_labels.append(int(load_level) * 3)
            pem_means.append(Pem)
            ps_means.append(Ps)
    if pem_means and ps_means:
        # Calculate deltas to 0-load
        pem0 = pem_means[0]
        ps0 = ps_means[0]
        pem_deltas = [v - pem0 for v in pem_means]
        ps_deltas = [v - ps0 for v in ps_means]
        plt.figure(figsize=(10,6))
        plt.plot(load_labels, pem_deltas, marker='o', label=r'$\Delta P_{EM}$')
        plt.plot(load_labels, ps_deltas, marker='s', label=r'$\Delta P_{S}$')
        plt.xlabel('Load Level (T/s)', fontsize=16)
        plt.ylabel('$\Delta P$ (W)', fontsize=16)
        #plt.title('$P\Delta$ vs. Load Level', fontsize=18)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(load_labels)
        plt.tight_layout()
        plt.savefig('./delta_power_vs_loadlevel.pdf', format='pdf', bbox_inches='tight')
        plt.close()
        print("Saved delta power line plot to ./delta_power_vs_loadlevel.pdf")


# Run the script if executed directly
if __name__ == "__main__":
    main()
