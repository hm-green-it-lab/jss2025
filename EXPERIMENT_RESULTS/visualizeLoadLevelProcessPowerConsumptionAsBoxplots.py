"""
visualizeLoadLevelProcessPowerConsumptionAsBoxplots.py

Estimate per-process power consumption and generate boxplots grouped by load level. This script contains helpers to parse procfs, powercap and other experiment logs and uses attribution models to distribute system-level power to processes (e.g., using procfs CPU fractions) and to add memory/storage power contributions.

Constants
- MEMORY_POWER_W_PER_GB, NETWORK_POWER_W_PER_GB, STORAGE_POWER_W_PER_TB are used as conversion factors for memory, network and storage demands.
"""

# Global constants for OTJAE power calculations
MEMORY_POWER_W_PER_GB = 0.392
NETWORK_POWER_W_PER_GB = 1.0
STORAGE_POWER_W_PER_TB = 1.2

def parse_kepler_http_logger(file_path, service_pids, trim_seconds=0, jmeter_bounds=None):
    """
    Parses a large http_logger_spring_docker_kepler file, extracts kepler_process_cpu_watts for the given service_pids.
    Only includes values within the jmeter_bounds timeframe if provided.
    Returns a DataFrame with columns: ['datetime', 'Power']
    """
    import re
    data = []
    current_timestamp = None
    # Regex for DATA line and kepler metric line
    data_line_re = re.compile(r"^DATA:.* at (\d+)")
    kepler_line_re = re.compile(r'kepler_process_cpu_watts\{[^}]*pid="(\d+)"[^}]*\} ([\d\.eE+-]+)')
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = data_line_re.match(line)
            if m:
                current_timestamp = int(m.group(1))
                continue
            if current_timestamp is not None:
                km = kepler_line_re.match(line)
                if km:
                    pid, value = km.group(1), km.group(2)
                    if pid in service_pids:
                        dt = pd.to_datetime(current_timestamp, unit='ms')
                        data.append({'datetime': dt, 'Power': float(value)})
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

def parse_procfs_joularjx(procfs_file, powercap_files, service_pids, trim_seconds=0, jmeter_bounds=None, n_cores=80, ticks_per_sec=100):
    #print(f"\n[DEBUG] --- parse_procfs_joularjx ---")
    #print(f"procfs_file: {procfs_file}")
    #print(f"powercap_files: {powercap_files}")
    #print(f"service_pids: {service_pids}")
    """
    Parses procfs CSV and powercap CSVs, computes process-specific power consumption over time.
    Returns a DataFrame with columns: ['datetime', 'Power'] for the process.
    """
    import pandas as pd
    # Factor out procfs parsing to reusable function
    proc_util, sys_df, _, _ = parse_procfs_data(procfs_file, service_pids, n_cores=n_cores, ticks_per_sec=ticks_per_sec, jmeter_bounds=jmeter_bounds)
    if proc_util is None or sys_df is None:
        return pd.DataFrame(columns=['datetime', 'Power'])
    # Read and sum powercap power for both sockets at each timestamp
    powercap_power_list = []
    for f in powercap_files:
        try:
            df_power = pd.read_csv(f)
            df_power = calculate_power_from_energy(df_power)
            if 'Power' in df_power.columns:
                powercap_power_list.append(df_power[['datetime', 'Power']])
            else:
                print(f"Power column not found after conversion in file: {f}")
        except Exception as e:
            print(f"Error processing powercap file {f}: {e}")
    if not powercap_power_list:
        print("No valid powercap power data found for JoularJX scenario.")
        return pd.DataFrame(columns=['datetime', 'Power'])
    powercap_df = pd.concat(powercap_power_list, ignore_index=True)
    powercap_df = powercap_df.groupby('datetime')['Power'].sum().reset_index()
    #print(f"powercap_df shape: {powercap_df.shape}")
    # Harmonize timestamps to seconds relative to steady-state start
    if jmeter_bounds is not None and jmeter_bounds[0] is not None:
        steady_state_start = jmeter_bounds[0]
    else:
        # fallback: use min timestamp in proc_util
        steady_state_start = proc_util['datetime'].min() if not proc_util.empty else None
    if steady_state_start is None:
        print("No steady-state start time found for harmonization.")
        return pd.DataFrame(columns=['datetime', 'Power'])
    # Add rel_sec column (integer seconds since steady-state start)
    proc_util = proc_util.copy()
    proc_util['rel_sec'] = (proc_util['datetime'] - steady_state_start).dt.total_seconds().astype(int)
    proc_util = proc_util[proc_util['rel_sec'] >= 0]
    #print(f"proc_util rel_sec min: {proc_util['rel_sec'].min() if not proc_util.empty else 'empty'}, max: {proc_util['rel_sec'].max() if not proc_util.empty else 'empty'}")
    powercap_df = powercap_df.copy()
    powercap_df['rel_sec'] = (powercap_df['datetime'] - steady_state_start).dt.total_seconds().astype(int)
    powercap_df = powercap_df[powercap_df['rel_sec'] >= 0]
    #print(f"powercap_df rel_sec min: {powercap_df['rel_sec'].min() if not powercap_df.empty else 'empty'}, max: {powercap_df['rel_sec'].max() if not powercap_df.empty else 'empty'}")
    # Merge on rel_sec (exact match)
    merged = pd.merge(proc_util, powercap_df, on='rel_sec', how='inner', suffixes=('_proc', '_powercap'))
    #print(f"merged shape: {merged.shape}")
    merged['Power'] = merged['util_ratio'] * merged['Power']
    # Use the datetime from proc_util for output
    result = merged[['datetime_proc', 'Power']].rename(columns={'datetime_proc': 'datetime'}).dropna()
    # Optionally trim
    if not result.empty and (trim_seconds > 0 or jmeter_bounds is not None):
        result = trim_time_series(result, trim_seconds, jmeter_bounds)
    return result


# New helper function to parse procfs file and return process and system CPU utilization values
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
        proc_power = []
        for pid in service_pids:
            pid_df = proc_df[proc_df['pid'] == pid][['datetime', 'userTime_s', 'systemTime_s']].copy()
            pid_df['total_cpu'] = pid_df['userTime_s'] + pid_df['systemTime_s']
            pid_df['delta_cpu'] = pid_df['total_cpu'].diff()
            pid_df['interval'] = pid_df['datetime'].diff().dt.total_seconds()
            pid_df = pid_df.iloc[1:]
            merged = pd.merge_asof(pid_df.sort_values('datetime'), sys_df.sort_values('datetime'), on='datetime', suffixes=('_proc', '_sys'), direction='nearest', tolerance=pd.Timedelta('1s'))
            merged['util_ratio'] = (merged['delta_cpu_proc'] / (merged['interval_proc'] * n_cores)) / (merged['delta_cpu_sys'] / (merged['interval_sys'] * n_cores))
            merged['util_ratio'] = merged['util_ratio'].clip(lower=0, upper=1)
            merged['pid'] = pid
            proc_power.append(merged[['datetime', 'pid', 'util_ratio']])
        if proc_power:
            proc_util = pd.concat(proc_power).groupby('datetime')['util_ratio'].sum().reset_index()
        else:
            proc_util = None
        # --- Memory (status) ---
        status_mask = df['SourceFile'].str.endswith(r'status')
        status_df = df[status_mask].copy()
        status_df['pid'] = status_df['SourceFile'].str.extract(r'/proc/(\d+)/status')
        status_df['datetime'] = pd.to_datetime(status_df['Timestamp'], unit='ms')
        mem_deltas = []
        for pid in service_pids:
            pid_status = status_df[status_df['pid'] == pid][['datetime', 'VmSize', 'VmRSS']].copy()
            pid_status = pid_status.sort_values('datetime')
            pid_status['delta_vmsize'] = pid_status['VmSize'].diff()
            pid_status['VmSize'] = pid_status['VmSize']
            pid_status['VmRSS'] = pid_status['VmRSS']
            pid_status['pid'] = pid
            mem_deltas.append(pid_status[['datetime', 'pid', 'VmSize', 'VmRSS']])
        if mem_deltas:
            mem_deltas_df = pd.concat(mem_deltas, ignore_index=True)
        else:
            mem_deltas_df = None
        # --- Storage (io) ---
        io_mask = df['SourceFile'].str.endswith(r'io')
        io_df = df[io_mask].copy()
        io_df['pid'] = io_df['SourceFile'].str.extract(r'/proc/(\d+)/io')
        io_df['datetime'] = pd.to_datetime(io_df['Timestamp'], unit='ms')
        io_deltas = []
        for pid in service_pids:
            pid_io = io_df[io_df['pid'] == pid][['datetime', 'read_bytes', 'write_bytes']].copy()
            pid_io = pid_io.sort_values('datetime')
            pid_io['delta_read_bytes'] = pid_io['read_bytes'].diff()
            pid_io['delta_write_bytes'] = pid_io['write_bytes'].diff()
            pid_io = pid_io.iloc[1:]
            pid_io['pid'] = pid
            io_deltas.append(pid_io[['datetime', 'pid', 'delta_read_bytes', 'delta_write_bytes']])
        if io_deltas:
            io_deltas_df = pd.concat(io_deltas, ignore_index=True)
        else:
            io_deltas_df = None
        return proc_util, sys_df, mem_deltas_df, io_deltas_df
    except Exception as e:
        print(f"Error in parse_procfs_data: {e}")
        return None, None, None, None

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
        #print(f"Reading file: {jtl_file}")
        df = pd.read_csv(jtl_file, low_memory=True)
        #print(df.dtypes)
        #col3, col4 = df.columns[3], df.columns[4]
        # Print rows where col3 or col4 is not a number (float or int)
        #bad_rows = df[~df[col3].apply(lambda x: isinstance(x, (int, float))) | ~df[col4].apply(lambda x: isinstance(x, (int, float)))]
        #print("Rows with mixed types in columns 3 or 4:")
        #print(bad_rows)
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


# Helper to process docker_otjae scenario
def process_docker_otjae(scenario_dir, trim_seconds, pcpumin, pcpumax):
    # Find jmeter bounds
    jmeter_bounds = get_jmeter_time_bounds(str(scenario_dir), trim_seconds)
    # Find procfs file
    procfs_files = list(scenario_dir.glob('**/procfs_spring_docker_otjae_*.csv'))
    if not procfs_files:
        return None
    procfs_file = str(procfs_files[0])
    # Find service_pids
    experiment_log_path = scenario_dir / 'logs' / 'experiment_log.jsonl'
    service_pids = extract_service_pids(str(experiment_log_path)) if experiment_log_path.exists() else []
    if not service_pids:
        return None
    # Parse procfs data (get system utilization)
    proc_util, sys_df, mem_deltas_df, io_deltas_df = parse_procfs_data(procfs_file, service_pids, jmeter_bounds=jmeter_bounds)
    if sys_df is None or sys_df.empty:
        return None
    # Calculate system CPU utilization per second
    sys_df = sys_df.copy()
    sys_df['sys_util'] = sys_df['delta_cpu'] / (sys_df['interval'])
    # Normalize to [0,1] by dividing by n_cores (assume 80 as default)
    n_cores = 80
    sys_df['sys_util_norm'] = sys_df['sys_util'] / n_cores
    sys_df['sys_util_norm'] = sys_df['sys_util_norm'].clip(lower=0, upper=1)
    # Calculate power: P = pcpumin + (sys_util_norm * (pcpumax - pcpumin))
    sys_df['Power'] = pcpumin + (sys_df['sys_util_norm'] * (pcpumax - pcpumin))


    # Add memory power (VmRSS in kB to GB, then * MEMORY_POWER_W_PER_GB)
    if mem_deltas_df is not None and not mem_deltas_df.empty:
        mem_group = mem_deltas_df.groupby('datetime')['VmRSS'].sum().reset_index()
        mem_group['VmRSS_GB'] = mem_group['VmRSS'] / (1024 * 1024)
        mem_group['Pmemory'] = mem_group['VmRSS_GB'] * MEMORY_POWER_W_PER_GB
        sys_df = pd.merge_asof(sys_df.sort_values('datetime'), mem_group[['datetime', 'Pmemory']].sort_values('datetime'), on='datetime', direction='nearest', tolerance=pd.Timedelta('1s'))
        sys_df['Pmemory'] = sys_df['Pmemory'].fillna(0)
        sys_df['Power'] = sys_df['Power'] + sys_df['Pmemory']

    # Add storage power (delta_read_bytes + delta_write_bytes in TB * STORAGE_POWER_W_PER_TB)
    if io_deltas_df is not None and not io_deltas_df.empty:
        # For each timestamp, sum deltas across all pids
        io_group = io_deltas_df.groupby('datetime')[['delta_read_bytes', 'delta_write_bytes']].sum().reset_index()
        # Convert bytes to TB
        io_group['total_bytes'] = io_group['delta_read_bytes'].fillna(0) + io_group['delta_write_bytes'].fillna(0)
        io_group['total_TB'] = io_group['total_bytes'] / (1024 ** 4)
        io_group['Pstorage'] = io_group['total_TB'] * STORAGE_POWER_W_PER_TB
        # Merge storage power into sys_df by datetime (nearest)
        sys_df = pd.merge_asof(sys_df.sort_values('datetime'), io_group[['datetime', 'Pstorage']].sort_values('datetime'), on='datetime', direction='nearest', tolerance=pd.Timedelta('1s'))
        sys_df['Pstorage'] = sys_df['Pstorage'].fillna(0)
        sys_df['Power'] = sys_df['Power'] + sys_df['Pstorage']

    # We do not add network power as we do not have per-process network I/O data

    # Return as a Series for boxplot
    return sys_df[['Power']].dropna()['Power']

def collect_data_by_load_level(trim_seconds=0, scenario_suffixes=None, included_load_levels=None):
    """
    Collects and groups all Rittal and Powercap data by load level (numeric prefix of directory name).
    Aggregates all runs (e.g., 350, 350_run2, 350_run3) for each load level.
    Only includes scenario subdirectories matching scenario_suffixes if provided.
    If included_load_levels is provided (list of strings), only those load levels are included in the returned data (except for pcpumin/pcpumax, which always use all data).
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
    # Prepare data structures
    data_by_load = defaultdict(lambda: defaultdict(lambda: {'rittal': [], 'powercap': [], 'kepler': [], 'scaphandre': [], 'joularjx': []}))
    # --- Step 1: Process docker_tools scenario first and collect mean powercap for 0 and 560 ---
    pcpumin = None
    pcpumax = None
    for load_level in ['0', '560']:
        dirs = load_level_map.get(load_level, [])
        powercap_means = []
        for run_path in dirs:
            for scenario_dir in run_path.iterdir():
                if not scenario_dir.is_dir():
                    continue
                scenario_name = scenario_dir.name
                if not scenario_name.endswith('docker_tools'):
                    continue
                jmeter_bounds = get_jmeter_time_bounds(str(scenario_dir), trim_seconds)
                powercap_files = list(scenario_dir.glob('**/powercap_*.csv'))
                for file_path in powercap_files:
                    try:
                        power_data = load_power_data(str(file_path), trim_seconds, jmeter_bounds)
                        if 'Power' in power_data.columns:
                            powercap_means.append(power_data['Power'].mean())
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        if powercap_means:
            mean_val = float(np.mean(powercap_means))
            if load_level == '0':
                pcpumin = mean_val
            elif load_level == '560':
                pcpumax = mean_val
    # Attach pcpumin and pcpumax to the result for later use
    data_by_load['pcpumin'] = pcpumin
    data_by_load['pcpumax'] = pcpumax
    print(f"Min Mean CPU Power: {pcpumin} W, Max Mean CPU Power: {pcpumax} W")
    # --- Step 2: Process all scenarios as before ---
    for load_level, dirs in load_level_map.items():
        # If included_load_levels is set, skip load levels not in the list (except for pcpumin/pcpumax)
        if included_load_levels is not None and load_level not in included_load_levels:
            continue
        for run_path in dirs:
            for scenario_dir in run_path.iterdir():
                if not scenario_dir.is_dir():
                    continue
                scenario_name = scenario_dir.name
                # Filter for scenario suffixes if provided
                if scenario_suffixes is not None and not any(scenario_name.endswith(suf) for suf in scenario_suffixes):
                    continue
                jmeter_bounds = get_jmeter_time_bounds(str(scenario_dir), trim_seconds)
                # Load and append docker_joularjx process power
                if scenario_name.endswith('docker_joularjx'):
                    procfs_files = list(scenario_dir.glob('**/procfs_spring_docker_joularjx_*.csv'))
                    powercap_files = list(scenario_dir.glob('**/powercap_*.csv'))
                    experiment_log_path = scenario_dir / 'logs' / 'experiment_log.jsonl'
                    service_pids = extract_service_pids(str(experiment_log_path)) if experiment_log_path.exists() else []
                    if procfs_files and powercap_files and service_pids:
                        try:
                            joularjx_data = parse_procfs_joularjx(str(procfs_files[0]), [str(f) for f in powercap_files], service_pids, trim_seconds, jmeter_bounds)
                            if not joularjx_data.empty:
                                data_by_load[load_level][scenario_name]['joularjx'].append(joularjx_data['Power'])
                        except Exception as e:
                            print(f"Error loading docker_joularjx procfs/powercap: {e}")
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
                # Load and append OTJAE data from procfs and jmeter
                if scenario_name.endswith('docker_otjae'):
                    pcpumin = data_by_load.get('pcpumin', None)
                    pcpumax = data_by_load.get('pcpumax', None)
                    if pcpumin is not None and pcpumax is not None:
                        try:
                            otjae_power = process_docker_otjae(scenario_dir, trim_seconds, pcpumin, pcpumax)
                            if otjae_power is not None and not otjae_power.empty:
                                data_by_load[load_level][scenario_name]['otjae'] = [otjae_power]
                        except Exception as e:
                            print(f"Error loading docker_otjae procfs/jmeter: {e}")
    return data_by_load


def plot_all_load_levels_in_one_row(data_by_load, output_path, scenario_suffixes=None, custom_labels=None, show_rittal=True):
    """
    Plots a single row of boxplots, one for each load level, each showing Rittal and Powercap data for selected scenarios.
    Groups all data for each scenario (across all runs) into a single boxplot per scenario per load level.
    scenario_suffixes: list of scenario suffixes to include and order.
    custom_labels: dict mapping scenario suffix to label.
    show_rittal: bool, whether to show Rittal (Pem) values or not.
    """
    # Only keep numeric load level keys (exclude e.g. 'pcpumin', 'pcpumax')
    numeric_items = [(k, v) for k, v in data_by_load.items() if k.isdigit()]
    sorted_loads = sorted(numeric_items, key=lambda x: int(x[0]))
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
            all_joularjx = []
            all_otjae = []
            for scenario in matching:
                all_rittal.extend(scenario_dict[scenario]['rittal'])
                all_powercap.extend(scenario_dict[scenario]['powercap'])
                all_kepler.extend(scenario_dict[scenario]['kepler'])
                all_scaphandre.extend(scenario_dict[scenario]['scaphandre'])
                all_joularjx.extend(scenario_dict[scenario]['joularjx'])
                if 'otjae' in scenario_dict[scenario]:
                    all_otjae.extend(scenario_dict[scenario]['otjae'])
            # Combine all Rittal data for this scenario (controlled by show_rittal)
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
                box_labels.append(f'{label}$P_{{P_{{K}}}}$')
            # Combine all Scaphandre data for this scenario
            if all_scaphandre:
                combined_scaphandre = pd.concat(all_scaphandre, ignore_index=True)
                label = custom_labels.get(suffix, suffix) if custom_labels else suffix
                box_data.append(combined_scaphandre)
                box_labels.append(f'{label}$P_{{P_{{Sc}}}}$')
            # Combine all JoularJX data for this scenario
            if all_joularjx:
                combined_joularjx = pd.concat(all_joularjx, ignore_index=True)
                label = custom_labels.get(suffix, suffix) if custom_labels else suffix
                box_data.append(combined_joularjx)
                box_labels.append(f'{label}$P_{{P_{{J}}}}$')
            # Combine all OTJAE data for this scenario
            if all_otjae:
                combined_otjae = pd.concat(all_otjae, ignore_index=True)
                label = custom_labels.get(suffix, suffix) if custom_labels else suffix
                box_data.append(combined_otjae)
                box_labels.append(f'{label}$P_{{P_{{O}}}}$')
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
        "docker_kepler": "",
        "docker_scaphandre": "",
        "docker_otjae": "",
        "docker_joularjx": ""
    }
    included_load_levels = ["230", "350", "480", "560"]  # Set which load levels to include
    data_by_load = collect_data_by_load_level(trim_seconds=trim_seconds, scenario_suffixes=scenario_suffixes, included_load_levels=included_load_levels)
    # Set show_rittal to True to show Rittal (Pem) values, False to hide
    show_rittal = False
    plot_all_load_levels_in_one_row(data_by_load, "./process_power_consumption_boxplots_all_loads.pdf", scenario_suffixes=scenario_suffixes, custom_labels=custom_labels, show_rittal=show_rittal)

    # --- LaTeX Table Generation ---
    # Only keep numeric load level keys (exclude e.g. 'pcpumin', 'pcpumax')
    numeric_items = [(k, v) for k, v in data_by_load.items() if k.isdigit()]
    sorted_loads = sorted(numeric_items, key=lambda x: int(x[0]))
    # Table header
    print("\\begin{table}")
    print("    \\centering")
    print("    \\resizebox{\\textwidth}{!}{%")
    print("        \\begin{tabular}{ |c|c|c|c|c|c|c|c|c|c| }")
    print("            \\hline")
    print("            \\multirow{2}{*}{Load} & \\multirow{2}{*}{P\\textsubscript{S}} & \\multicolumn{2}{|c|}{Kepler}  & \\multicolumn{2}{|c|}{Scaphandre}& \\multicolumn{2}{|c|}{OTJAE}  & \\multicolumn{2}{|c|}{JoularJX} \\\\")
    print("            \\cline{3-10}")
    print("            &          & P\\textsubscript{P} &   $\\Delta$    & P\\textsubscript{P}    &  $\\Delta$   & P\\textsubscript{P} & $\\Delta$ & P\\textsubscript{P} & $\\Delta$ \\\\")
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
        # Kepler (Ppk)
        Ppk = get_mean_power('docker_kepler', 'kepler')
        # Scaphandre (Pps)
        Pps = get_mean_power('docker_scaphandre', 'scaphandre')
        # OTJAE (Ppo)
        Ppo = get_mean_power('docker_otjae', 'otjae')
        # JoularJX (Ppj)
        Ppj = get_mean_power('docker_joularjx', 'joularjx')
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
        print(f"            {load_label} & {fmt(Ps)}  & {fmt(Ppk)}  & {pct(Ppk, Ps)} & {fmt(Pps)} & {pct(Pps, Ps)} & {fmt(Ppo)} & {pct(Ppo, Ps)} & {fmt(Ppj)} & {pct(Ppj, Ps)} \\\\")
        print("            \\hline")
    print("        \\end{tabular}")
    print("    }")
    print("    \\caption{Mean process power consumption by load level}")
    print("    \\label{tab:process_power_depending_throughput}")
    print("\\end{table}")


# Run the script if executed directly
if __name__ == "__main__":
    main()
