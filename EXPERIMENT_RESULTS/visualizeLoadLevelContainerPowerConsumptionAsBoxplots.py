"""
visualizeLoadLevelContainerPowerConsumptionAsBoxplots.py

Collect container-level power measurements (container metrics like Kepler, Scaphandre or aggregated package-level RAPL) and produce per-load boxplots.

The module contains parsers for specialized HTTP logger metric files as well as helpers to aggregate Rittal and powercap inputs. It can also print LaTeX tables summarizing container power by load.
"""

def parse_kepler_http_logger(file_path, service_pids, trim_seconds=0, jmeter_bounds=None):
    """
    Parses a large http_logger_spring_docker_kepler file, extracts kepler_process_cpu_watts for the given service_pids.
    Only includes values within the jmeter_bounds timeframe if provided.
    Returns a DataFrame with columns: ['datetime', 'Power']
    """
    import re
    data = []
    current_timestamp = None
    # Regex for DATA line and kepler metric lines
    data_line_re = re.compile(r"^DATA:.* at (\d+)")
    kepler_proc_re = re.compile(r'kepler_process_cpu_watts\{([^}]*)\} ([\d\.eE+-]+)')
    kepler_cont_re = re.compile(r'kepler_container_cpu_watts\{([^}]*)\} ([\d\.eE+-]+)')
    # Helper to parse label string into dict
    def parse_labels(label_str):
        return dict(re.findall(r'(\w+)="([^"]*)"', label_str))
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = list(f)
    i = 0
    while i < len(lines):
        m = data_line_re.match(lines[i])
        if m:
            current_timestamp = int(m.group(1))
            # Collect all metrics for this timestamp
            i += 1
            proc_metrics = []
            cont_metrics = []
            while i < len(lines) and not data_line_re.match(lines[i]):
                km = kepler_proc_re.match(lines[i])
                if km:
                    labels = parse_labels(km.group(1))
                    value = km.group(2)
                    proc_metrics.append((labels, value))
                else:
                    kc = kepler_cont_re.match(lines[i])
                    if kc:
                        labels = parse_labels(kc.group(1))
                        value = kc.group(2)
                        cont_metrics.append((labels, value))
                i += 1
            # For each process metric, if pid matches and container_id present, use container metric if available
            for labels, value in proc_metrics:
                pid = labels.get('pid')
                container_id = labels.get('container_id')
                if pid in service_pids:
                    dt = pd.to_datetime(current_timestamp, unit='ms')
                    if container_id:
                        # Find matching container metric
                        cont_val = None
                        for clabels, cvalue in cont_metrics:
                            if clabels.get('container_id') == container_id:
                                cont_val = cvalue
                                break
                        if cont_val is not None:
                            data.append({'datetime': dt, 'Power': float(cont_val)})
                        else:
                            data.append({'datetime': dt, 'Power': float(value)})
                    else:
                        data.append({'datetime': dt, 'Power': float(value)})
        else:
            i += 1
    df = pd.DataFrame(data)
    # Optionally trim the time series
    if not df.empty and (trim_seconds > 0 or jmeter_bounds is not None):
        df = trim_time_series(df, trim_seconds, jmeter_bounds)
    return df

def parse_scaphandre_http_logger(file_path, service_pids, trim_seconds=0, jmeter_bounds=None):
    """
    Parses a large http_logger_spring_docker_scaphandre file, extracts scaph_process_power_consumption_microwatts for the given service_pids.
    Converts value to watts. Only includes values within the jmeter_bounds timeframe if provided.
    Returns a DataFrame with columns: ['datetime', 'Power']
    """
    import re
    data = []
    current_timestamp = None
    # Regex for DATA line and scaphandre metric line
    data_line_re = re.compile(r"^DATA:.* at (\d+)")
    scaphandre_line_re = re.compile(r'scaph_process_power_consumption_microwatts\{[^}]*pid="(\d+)"[^}]*\} ([\d\.eE+-]+)')
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = data_line_re.match(line)
            if m:
                current_timestamp = int(m.group(1))
                continue
            if current_timestamp is not None:
                sm = scaphandre_line_re.match(line)
                if sm:
                    pid, value = sm.group(1), sm.group(2)
                    if pid in service_pids:
                        dt = pd.to_datetime(current_timestamp, unit='ms')
                        # Convert microwatts to watts
                        data.append({'datetime': dt, 'Power': float(value) / 1_000_000})
    df = pd.DataFrame(data)
    # Optionally trim the time series
    if not df.empty and (trim_seconds > 0 or jmeter_bounds is not None):
        df = trim_time_series(df, trim_seconds, jmeter_bounds)
    return df
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from collections import defaultdict
import os

import json

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


def collect_data_by_load_level(trim_seconds=0, scenario_suffixes=None, included_load_levels=None):
    """
    Collects and groups all Rittal and Powercap data by load level (numeric prefix of directory name).
    Aggregates all runs (e.g., 350, 350_run2, 350_run3) for each load level.
    Only includes scenario subdirectories matching scenario_suffixes if provided.
    If included_load_levels is provided (list of strings), only those load levels are included in the returned data.
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
    data_by_load = defaultdict(lambda: defaultdict(lambda: {'rittal': [], 'powercap': [], 'kepler': [], 'scaphandre': []}))
    for load_level, dirs in load_level_map.items():
        # If included_load_levels is set, skip load levels not in the list
        if included_load_levels is not None and load_level not in included_load_levels:
            continue
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
                # Only include Rittal and Powercap data if scenario_name ends with 'docker_tools'
                if scenario_name.endswith('docker_tools'):
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
                # Load and append Kepler data from http_logger_spring_docker_kepler files
                kepler_files = list(scenario_dir.glob('**/http_logger_spring_docker_kepler*.csv'))
                if kepler_files:
                    experiment_log_path = scenario_dir / 'logs' / 'experiment_log.jsonl'
                    service_pids = extract_service_pids(str(experiment_log_path)) if experiment_log_path.exists() else []
                    for file_path in kepler_files:
                        try:
                            kepler_data = parse_kepler_http_logger(str(file_path), service_pids, trim_seconds, jmeter_bounds)
                            if not kepler_data.empty:
                                data_by_load[load_level][scenario_name]['kepler'].append(kepler_data['Power'])
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                # Load and append Scaphandre data from http_logger_spring_docker_scaphandre files
                scaphandre_files = list(scenario_dir.glob('**/http_logger_spring_docker_scaphandre*.csv'))
                if scaphandre_files:
                    experiment_log_path = scenario_dir / 'logs' / 'experiment_log.jsonl'
                    service_pids = extract_service_pids(str(experiment_log_path)) if experiment_log_path.exists() else []
                    for file_path in scaphandre_files:
                        try:
                            scaphandre_data = parse_scaphandre_http_logger(str(file_path), service_pids, trim_seconds, jmeter_bounds)
                            if not scaphandre_data.empty:
                                data_by_load[load_level][scenario_name]['scaphandre'].append(scaphandre_data['Power'])
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
    return data_by_load


def plot_all_load_levels_in_one_row(data_by_load, output_path, scenario_suffixes=None, custom_labels=None, show_rittal=True):
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
            all_kepler = []
            all_scaphandre = []
            for scenario in matching:
                all_rittal.extend(scenario_dict[scenario]['rittal'])
                all_powercap.extend(scenario_dict[scenario]['powercap'])
                all_kepler.extend(scenario_dict[scenario]['kepler'])
                all_scaphandre.extend(scenario_dict[scenario]['scaphandre'])
            # Combine all Rittal data for this scenario
            if show_rittal and all_rittal:
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
            # Combine all Kepler data for this scenario
            if all_kepler:
                combined_kepler = pd.concat(all_kepler, ignore_index=True)
                label = custom_labels.get(suffix, suffix) if custom_labels else suffix
                box_data.append(combined_kepler)
                box_labels.append(f'{label}$P_{{C_{{K}}}}$')
            # Combine all Scaphandre data for this scenario
            if all_scaphandre:
                combined_scaphandre = pd.concat(all_scaphandre, ignore_index=True)
                label = custom_labels.get(suffix, suffix) if custom_labels else suffix
                box_data.append(combined_scaphandre)
                box_labels.append(f'{label}$P_{{C_{{Sc}}}}$')
        # Plot boxplot if data exists
        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, medianprops=dict(color="grey", linewidth=2.5),
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


# --- LaTeX Table Generation ---
    # Only keep numeric load level keys (exclude e.g. 'pcpumin', 'pcpumax')
    numeric_items = [(k, v) for k, v in data_by_load.items() if k.isdigit()]
    sorted_loads = sorted(numeric_items, key=lambda x: int(x[0]))
    print("\\begin{table}")
    print("    \\begin{center}")
    print("        \\begin{tabular}{ |c|c|c|c|c|c| }")
    print("            \\hline")
    print("            \\multirow{2}{*}{Load} & \\multirow{2}{*}{P\\textsubscript{S}}  & \\multicolumn{2}{|c|}{Kepler}  & \\multicolumn{2}{|c|}{Scaphandre} \\")
    print("            \\cline{3-6}")
    print("            &           & P\\textsubscript{C} &  $\\Delta$ & P\\textsubscript{C} &  $\\Delta$ \\")
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
        # Powercap (Ps)
        Ps = get_mean_power('docker_tools', 'powercap')
        # Kepler (Pc)
        Pkc = get_mean_power('docker_kepler', 'kepler')
        # Scaphandre (Pc)
        Psc = get_mean_power('docker_scaphandre', 'scaphandre')
        # Calculate deltas (as percent of Ps)
        def pct(val, ref):
            if val is None or ref is None or ref == 0:
                return "-"
            return f"{(val / ref * 100):.2f}\\%"
        # Format values
        def fmt(val):
            return f"{val:.2f}W" if val is not None else "-"
        # Load label (convert to T/s)
        load_label = f"{int(load_level)*3}T/s"
        print(f"            {load_label} & {fmt(Ps)}  & {fmt(Pkc)}  & {pct(Pkc, Ps)} & {fmt(Psc)} & {pct(Psc, Ps)} \\")
        print("            \\hline")
    print("        \\end{tabular}")
    print("        \\caption{Mean container power consumption by load level}")
    print("        \\label{tab:container_power_depending_throughput}")
    print("    \\end{center}")
    print("\\end{table}")


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
        #"docker_otjae",
        #"docker_joularjx"
    ]
    # Optional: custom labels for scenario suffixes
    custom_labels = {
        "idle_no_tools": "Idle",
        "docker_tools": "",
        "docker_none": "Container\nIdle (CI)",
        "docker_idle": "CI, Powercap (PC)\nand ProcFS",
        "docker_kepler": "",
        "docker_scaphandre": "",
        "docker_otjae": "CI, PC, ProcFS\nand OTJAE",
        "docker_joularjx": "CI, PC, ProcFS\nand JoularJX"
    }
    included_load_levels = ["230", "350", "480", "560"]  # Set which load levels to include
    show_rittal = False  # Set to False to hide Rittal (Pem) values in the plot
    data_by_load = collect_data_by_load_level(trim_seconds=trim_seconds, scenario_suffixes=scenario_suffixes, included_load_levels=included_load_levels)
    plot_all_load_levels_in_one_row(data_by_load, "./container_power_consumption_boxplots_all_loads.pdf", scenario_suffixes=scenario_suffixes, custom_labels=custom_labels, show_rittal=show_rittal)


# Run the script if executed directly
if __name__ == "__main__":
    main()
