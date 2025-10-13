# Global constants for OTJAE power calculations
MEMORY_POWER_W_PER_GB = 0.392
NETWORK_POWER_W_PER_GB = 1.0
STORAGE_POWER_W_PER_TB = 1.2

 # --- Boxplot configuration ---
# Set to 'per_second' or 'per_invocation' to control which data is shown in the boxplots
BOXPLOT_DATA_MODE = 'per_invocation'  # Options: 'per_second', 'per_invocation'

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from collections import defaultdict
import os
import re
import json

def calculate_otjae_transaction_power_cpu(scenario_dir, procfs_file, service_pids, otjae_per_second, jmeter_bounds=None, n_cores=80, ticks_per_sec=100, data_by_load=None):
    """
    Calculate per-transaction power for OTJAE using CPU time fraction per second and process power from procfs.
    otjae_per_second: output of parse_otjae_transaction_resource (per-second resource demand)
    Returns: dict {transaction: [P_transaction_per_sec, ...]}
    """
    # Use process_docker_otjae to get process power per second (includes CPU, memory, storage)
    # We need pcpumin and pcpumax, but for this function, we assume they are available from the main data collection (they can be passed as args if needed)
    # For now, try to extract them from the global data_by_load if available, else fallback to None
    pcpumin = data_by_load.get('pcpumin', None)
    pcpumax = data_by_load.get('pcpumax', None)
    if pcpumin is None or pcpumax is None:
        # fallback: try to estimate from process power min/max
        pcpumin = 0
        pcpumax = 1
    # Get process power time series (indexed by datetime)
    process_power_series = process_docker_otjae(scenario_dir, 0, pcpumin, pcpumax)
    if process_power_series is None or process_power_series.empty:
        return {}
    # Build mapping from second (int) to process power (float)
    process_power_sec = {}
    for sec, val in process_power_series.items():
        process_power_sec[sec] = val

    # Harmonize seconds: find the minimum second in otjae_per_second and cpu_time_per_sec, shift all to start from 0
    otjae_seconds = list(otjae_per_second.keys())
    # Get process CPU time per second (from sys_df)
    proc_util, sys_df, _, _ = parse_procfs_data(procfs_file, service_pids, n_cores=n_cores, ticks_per_sec=ticks_per_sec, jmeter_bounds=jmeter_bounds)
    if sys_df is None or sys_df.empty:
        return {}
    sys_df = sys_df.copy()
    sys_df['sec'] = (sys_df['datetime'].astype('int64') // 10**9).astype(int)
    cpu_time_per_sec = sys_df.groupby('sec')['delta_cpu'].sum()
    cpu_seconds = list(cpu_time_per_sec.index)
    # Find the minimum second across both otjae_per_second and cpu_time_per_sec
    all_seconds = otjae_seconds + cpu_seconds
    if not all_seconds:
        return {}
    min_sec = min(all_seconds)
    # Shift otjae_per_second keys
    otjae_per_second_rel = {sec - min_sec: txs for sec, txs in otjae_per_second.items()}
    # Shift cpu_time_per_sec keys
    cpu_time_per_sec_rel = {sec - min_sec: val for sec, val in cpu_time_per_sec.items()}

    # Build per-transaction power per second, normalized by number of invocations and also total per second
    tx_power_per_invocation = {}
    tx_power_per_second = {}
    for sec, txs in otjae_per_second_rel.items():
        proc_cpu = cpu_time_per_sec_rel.get(sec, None)
        P_proc = process_power_sec.get(sec, None)
        if proc_cpu is None or proc_cpu == 0 or P_proc is None:
            continue
        for tx, vals in txs.items():
            cpu_tx = vals['cpu']
            mem_tx = vals.get('mem', 0)
            net_tx = vals.get('net', 0)
            disk_tx = vals.get('disk', 0)
            num_invocations = vals.get('count', 0)
            # Convert cpu_tx from nanoseconds to seconds
            cpu_tx_sec = cpu_tx / 1e9
            # Convert mem_tx (bytes) to GB
            mem_tx_gb = mem_tx / (1024 ** 3)
            # Convert net_tx (bytes) to GB
            net_tx_gb = net_tx / (1024 ** 3)
            # Convert disk_tx (bytes) to TB
            disk_tx_tb = disk_tx / (1024 ** 4)
            # Calculate total transaction power for this second (all invocations)
            if proc_cpu > 0:
                P_tx_cpu_total = (cpu_tx_sec / proc_cpu) * P_proc
            else:
                P_tx_cpu_total = 0
            P_tx_mem_total = mem_tx_gb * MEMORY_POWER_W_PER_GB
            P_tx_net_total = net_tx_gb * NETWORK_POWER_W_PER_GB
            P_tx_disk_total = disk_tx_tb * STORAGE_POWER_W_PER_TB
            P_tx_total = P_tx_cpu_total + P_tx_mem_total + P_tx_net_total + P_tx_disk_total
            # Calculate per-invocation value (for boxplots)
            if num_invocations > 0:
                P_tx = P_tx_total / num_invocations
            else:
                P_tx = 0
            if tx not in tx_power_per_invocation:
                tx_power_per_invocation[tx] = []
            if tx not in tx_power_per_second:
                tx_power_per_second[tx] = []
            tx_power_per_invocation[tx].append(P_tx)
            tx_power_per_second[tx].append(P_tx_total)
    return {'per_invocation': tx_power_per_invocation, 'per_second': tx_power_per_second}

def parse_otjae_transaction_resource(log_file, jmeter_bounds=None):
    """
    Parse OTJAE docker_compose_logs_*.txt file for transaction resource demand, filtering by JMeter steady-state.
    Returns a dict: {transaction: {'cpu': ..., 'mem': ..., 'net': ..., 'disk': ..., 'count': ...}}
    """
    from collections import defaultdict
    import pandas as pd
    # Get bounds in ms since epoch
    if jmeter_bounds is not None and all(jmeter_bounds):
        start, end = jmeter_bounds
        start_ms = int(start.value // 10**6)
        end_ms = int(end.value // 10**6)
    else:
        start_ms = end_ms = None
    per_second = defaultdict(lambda: defaultdict(lambda: {'cpu': 0, 'mem': 0, 'net': 0, 'disk': 0, 'count': 0}))

    # Compile regex patterns outside the loop
    logsystemtime_pattern = re.compile(r'io\.retit\.logsystemtime=([^,}}]+)')
    startthread_pattern = re.compile(r'io\.retit\.startthread=([^,}}]+)')
    endthread_pattern = re.compile(r'io\.retit\.endthread=([^,}}]+)')
    server_span_pattern = re.compile(r'\b[\da-f]{16} SERVER \[tracer:')
    keyval_pattern = re.compile(r'(\w[\w.]+)=([^,}}]+)')

    # Batch read all lines
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Batch filter relevant lines
    filtered_lines = [
        line for line in lines
        if ('io.retit.logsystemtime' in line and 'AttributesMap' in line and server_span_pattern.search(line))
    ]

    # Process only filtered lines
    for line in filtered_lines:
        # Extract logsystemtime
        m_logsystem = logsystemtime_pattern.search(line)
        if not m_logsystem:
            continue
        try:
            logsystemtime = int(m_logsystem.group(1))
        except Exception:
            continue
        if start_ms is not None and end_ms is not None:
            if not (start_ms <= logsystemtime <= end_ms):
                continue

        # Extract startthread and endthread
        m_startthread = startthread_pattern.search(line)
        m_endthread = endthread_pattern.search(line)
        if not m_startthread or not m_endthread:
            continue
        try:
            startthread = int(m_startthread.group(1))
            endthread = int(m_endthread.group(1))
        except Exception:
            continue
        if startthread + endthread == 0:
            continue
        if startthread != endthread:
            continue

        # Extract all key-value pairs (attributes)
        attributes = dict(keyval_pattern.findall(line))
        method = attributes.get('http.request.method')
        if not method:
            continue
        key = f"{method}"
        # Calculate deltas
        def calc_delta(attr, startk, endk):
            try:
                return max(int(attributes.get(endk, '0')) - int(attributes.get(startk, '0')), 0)
            except Exception:
                return 0
        cpu = calc_delta(attributes, 'io.retit.startcputime', 'io.retit.endcputime')
        mem = calc_delta(attributes, 'io.retit.startheapbyteallocation', 'io.retit.endheapbyteallocation')
        disk = calc_delta(attributes, 'io.retit.startdiskreaddemand', 'io.retit.enddiskreaddemand') + \
               calc_delta(attributes, 'io.retit.startdiskwritedemand', 'io.retit.enddiskwritedemand')
        net = calc_delta(attributes, 'io.retit.startnetworkreaddemand', 'io.retit.endnetworkreaddemand') + \
              calc_delta(attributes, 'io.retit.startnetworkwritedemand', 'io.retit.endnetworkwritedemand')
        # Use the second (rounded from ms) as the time bin
        sec = logsystemtime // 1000
        per_second[sec][key]['cpu'] += cpu
        per_second[sec][key]['mem'] += mem
        per_second[sec][key]['disk'] += disk
        per_second[sec][key]['net'] += net
        per_second[sec][key]['count'] += 1
    return per_second

def parse_joularjx_transaction_power(methods_dir, jmeter_bounds=None):
    """
    Parse all JoularJX methods-power.csv files in a directory, only using files whose timestamp is within jmeter_bounds.
    Returns a dict: {'per_invocation': {transaction_type: [...]}, 'per_second': {transaction_type: [...]}, 'total_energy': {transaction_type: total_energy}, 'steady_state_time': steady_state_time}
    """
    import re
    from collections import defaultdict
    import pandas as pd
    if not Path(methods_dir).is_dir():
        return {}

    # --- Validation: Read total energy file if available ---
    total_methods_dir = Path(methods_dir).parent.parent / "total" / "methods"
    total_energy_file = total_methods_dir / "joularJX-1-filtered-methods-energy.csv"
    total_energy_from_file = {}
    if total_energy_file.exists():
        try:
            import csv
            with open(total_energy_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) == 2:
                        method, energy = row
                        if method.endswith(".getData"):
                            total_energy_from_file["GET"] = float(energy)
                        elif method.endswith(".postData"):
                            total_energy_from_file["POST"] = float(energy)
                        elif method.endswith(".deleteData"):
                            total_energy_from_file["DELETE"] = float(energy)
        except Exception as e:
            print(f"[JoularJX Validation] Error reading total energy file: {e}")
    # Get bounds in ms since epoch
    if jmeter_bounds is not None and all(jmeter_bounds):
        start, end = jmeter_bounds
        start_ms = int(start.value // 10**6)
        end_ms = int(end.value // 10**6)
    else:
        start_ms = end_ms = None
    # Find all files matching pattern
    files = list(Path(methods_dir).glob('joularJX-*-filtered-methods-power.csv'))
    # Extract timestamp from filename
    file_tuples = []
    for f in files:
        m = re.search(r'joularJX-\d+-(\d+)-filtered-methods-power.csv', f.name)
        if m:
            ts = int(m.group(1))
            file_tuples.append((ts, f))
    # Filter files to steady-state
    if start_ms is not None and end_ms is not None:
        file_tuples = [(ts, f) for ts, f in file_tuples if start_ms <= ts <= end_ms]
    # For each file, calculate power per transaction type using time delta
    per_invocation = defaultdict(list)
    per_second = defaultdict(list)
    total_energy = defaultdict(float)
    prev_ts = None
    steady_state_start = None
    steady_state_end = None
    sorted_files = sorted(file_tuples)
    for idx, (ts, f) in enumerate(sorted_files):
        if steady_state_start is None or ts < steady_state_start:
            steady_state_start = ts
        if steady_state_end is None or ts > steady_state_end:
            steady_state_end = ts
        df = pd.read_csv(f, header=None, names=['method', 'energy'])
        # Determine time delta to previous file (in seconds)
        if prev_ts is not None:
            delta_t = (ts - prev_ts) / 1000.0  # timestamps are in ms
        else:
            # For the first file, if only one file, set delta_t=1, else use next file's delta
            if len(sorted_files) > 1 and idx+1 < len(sorted_files):
                next_ts = sorted_files[idx+1][0]
                delta_t = (next_ts - ts) / 1000.0
            else:
                delta_t = 1.
        #print(f"Processing file: {f}, ts: {ts}, delta_t: {delta_t}")        
        prev_ts = ts
        for ttype in ['getData', 'postData', 'deleteData']:
            mask = df['method'].str.endswith(f'.{ttype}')
            energy_sum = df.loc[mask, 'energy'].sum()
            num_invocations = mask.sum()
            # Power per second for this transaction type (all invocations in this second)
            if delta_t > 0:
                power_per_second = energy_sum / delta_t
            else:
                power_per_second = 0
            per_second[ttype.upper().replace('DATA','')].append(power_per_second)
            # Power per invocation (for boxplots)
            if delta_t > 0 and num_invocations > 0:
                power_per_invocation = (energy_sum / delta_t) / num_invocations
            else:
                power_per_invocation = 0
            per_invocation[ttype.upper().replace('DATA','')].append(power_per_invocation)
            # Sum total energy for this transaction type
            total_energy[ttype.upper().replace('DATA','')] += energy_sum

    # --- Validation: Compare calculated total_energy with file values and print results ---
    if total_energy_from_file:
        print("[JoularJX Validation] Comparing calculated total_energy with total file values:")
        for ttype in ['GET', 'POST', 'DELETE']:
            calc_val = total_energy.get(ttype, None)
            file_val = total_energy_from_file.get(ttype, None)
            if calc_val is not None and file_val is not None:
                diff = calc_val - file_val
                rel_diff = (diff / file_val) * 100 if file_val != 0 else float('inf')
                print(f"  {ttype}: calculated = {calc_val:.2f}, file = {file_val:.2f}, diff = {diff:.2f} ({rel_diff:+.2f}%)")
            elif calc_val is not None:
                print(f"  {ttype}: calculated = {calc_val:.2f}, file = MISSING")
            elif file_val is not None:
                print(f"  {ttype}: calculated = MISSING, file = {file_val:.2f}")
    else:
        print("[JoularJX Validation] No total energy file found for validation.")
    # Calculate steady-state time in seconds
    if steady_state_start is not None and steady_state_end is not None and steady_state_end > steady_state_start:
        steady_state_time = (steady_state_end - steady_state_start) / 1000.0
    else:
        steady_state_time = None
    return {
        'per_invocation': per_invocation,
        'per_second': per_second,
        'total_energy': total_energy,
        'steady_state_time': steady_state_time,
        'total_energy_from_file': total_energy_from_file
    }

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
    power_data['datetime'] = pd.to_datetime(power_data['Timestamp'].astype('int64'), unit='ms')
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
    data_by_load = defaultdict(lambda: defaultdict(lambda: {'rittal': [], 'powercap': [], 'joularjx_tx': {}, 'otjae_tx': {}}))
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
    import threading
    thread_list = []
    from threading import Lock
    data_lock = Lock()

    def process_scenario(load_level, scenario_dir):
        nonlocal data_by_load
        scenario_name = scenario_dir.name
        # Filter for scenario suffixes if provided
        if scenario_suffixes is not None and not any(scenario_name.endswith(suf) for suf in scenario_suffixes):
            return
        jmeter_bounds = get_jmeter_time_bounds(str(scenario_dir), trim_seconds)
        # Load and append docker_joularjx transaction-level power
        if scenario_name.endswith('docker_joularjx'):
            methods_dirs = list(scenario_dir.glob('**/app/runtime/methods'))
            if methods_dirs:
                tx_power = parse_joularjx_transaction_power(methods_dirs[0], jmeter_bounds)
                if tx_power:
                    with data_lock:
                        data_by_load[load_level][scenario_name]['joularjx_tx'] = tx_power
        if scenario_name.endswith('docker_tools'):
            rittal_files = list(scenario_dir.glob('**/rittal_*.csv'))
            powercap_files = list(scenario_dir.glob('**/powercap_*.csv'))
            # Load and append Rittal data
            for file_path in rittal_files:
                try:
                    power_data = load_rittal_data(str(file_path), trim_seconds, jmeter_bounds)
                    if 'Power (Watts)' in power_data.columns:
                        with data_lock:
                            data_by_load[load_level][scenario_name]['rittal'].append(power_data['Power (Watts)'])
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            # Load and append Powercap data
            for file_path in powercap_files:
                try:
                    power_data = load_power_data(str(file_path), trim_seconds, jmeter_bounds)
                    if 'Power' in power_data.columns:
                        with data_lock:
                            data_by_load[load_level][scenario_name]['powercap'].append(power_data['Power'])
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        # Load and append OTJAE per-transaction power (CPU)
        if scenario_name.endswith('docker_otjae'):
            log_files = list(scenario_dir.glob('**/docker_compose_logs_*.txt'))
            procfs_files = list(scenario_dir.glob('**/procfs_spring_docker_otjae_*.csv'))
            experiment_log_path = scenario_dir / 'logs' / 'experiment_log.jsonl'
            service_pids = extract_service_pids(str(experiment_log_path)) if experiment_log_path.exists() else []
            if log_files and procfs_files and service_pids:
                otjae_per_second = parse_otjae_transaction_resource(str(log_files[0]), jmeter_bounds)
                tx_power = calculate_otjae_transaction_power_cpu(scenario_dir, str(procfs_files[0]), service_pids, otjae_per_second, jmeter_bounds, data_by_load=data_by_load)
                if tx_power:
                    with data_lock:
                        data_by_load[load_level][scenario_name]['otjae_tx'] = tx_power
        print(f"[Thread finished] Scenario: {scenario_name}, Load Level: {load_level}")

    for load_level, dirs in load_level_map.items():
        # If included_load_levels is set, skip load levels not in the list (except for pcpumin/pcpumax)
        if included_load_levels is not None and load_level not in included_load_levels:
            continue
        for run_path in dirs:
            for scenario_dir in run_path.iterdir():
                if not scenario_dir.is_dir():
                    continue
                t = threading.Thread(target=process_scenario, args=(load_level, scenario_dir))
                thread_list.append(t)
                t.start()

    # Wait for all threads to finish
    for t in thread_list:
        t.join()
    return data_by_load


def plot_all_load_levels_in_one_row(data_by_load, output_path, scenario_suffixes=None, custom_labels=None):
    """
    Plots a single row of boxplots, one for each load level, each showing Rittal and Powercap data for selected scenarios.
    Groups all data for each scenario (across all runs) into a single boxplot per scenario per load level.
    scenario_suffixes: list of scenario suffixes to include and order.
    custom_labels: dict mapping scenario suffix to label.
    """
    # Only keep numeric load level keys (exclude e.g. 'pcpumin', 'pcpumax')
    numeric_items = [(k, v) for k, v in data_by_load.items() if k.isdigit()]
    sorted_loads = sorted(numeric_items, key=lambda x: int(x[0]))
    n_levels = len(sorted_loads)
    # Adjust width: 6 boxplots per load level, 3 load levels, make each level wider
    # E.g., 6 boxplots * 0.7 inch per box * 3 load levels = 12.6 inches, add some margin
    boxplots_per_level = 6
    n_loads = n_levels
    width_per_box = 0.7
    fig_width = max(12, boxplots_per_level * width_per_box * n_loads + 2)  # Ensure at least 12 inches, add margin
    fig, axes = plt.subplots(1, n_levels, figsize=(fig_width, 6), sharey=True)

    if n_levels == 1:
        axes = [axes]
    for ax, (load_level, scenario_dict) in zip(axes, sorted_loads):
        box_data = []
        box_labels = []
        # Aggregate all transaction data across all runs for this load level
        tx_agg_jx = {'GET': [], 'POST': [], 'DELETE': []}
        tx_agg_ot = {'GET': [], 'POST': [], 'DELETE': []}
        for scenario in scenario_dict:
            # JoularJX: use selected data mode if new structure, else fallback to old
            if 'joularjx_tx' in scenario_dict[scenario]:
                joularjx_tx = scenario_dict[scenario]['joularjx_tx']
                if isinstance(joularjx_tx, dict) and BOXPLOT_DATA_MODE in joularjx_tx:
                    tx_dict = joularjx_tx[BOXPLOT_DATA_MODE]
                elif isinstance(joularjx_tx, dict) and 'per_invocation' in joularjx_tx:
                    tx_dict = joularjx_tx['per_invocation']
                else:
                    tx_dict = joularjx_tx
                for ttype in ['GET', 'POST', 'DELETE']:
                    if ttype in tx_dict and tx_dict[ttype]:
                        tx_agg_jx[ttype].extend(tx_dict[ttype])
            # OTJAE: use selected data mode if new structure, else fallback to old
            if 'otjae_tx' in scenario_dict[scenario]:
                otjae_tx = scenario_dict[scenario]['otjae_tx']
                if isinstance(otjae_tx, dict) and BOXPLOT_DATA_MODE in otjae_tx:
                    tx_dict = otjae_tx[BOXPLOT_DATA_MODE]
                elif isinstance(otjae_tx, dict) and 'per_invocation' in otjae_tx:
                    tx_dict = otjae_tx['per_invocation']
                else:
                    tx_dict = otjae_tx
                for ttype in ['GET', 'POST', 'DELETE']:
                    if ttype in tx_dict and tx_dict[ttype]:
                        tx_agg_ot[ttype].extend(tx_dict[ttype])
        # Now add one boxplot per transaction type (if any data)
        for ttype in ['GET', 'POST', 'DELETE']:
            if tx_agg_jx[ttype]:
                box_data.append(pd.Series(tx_agg_jx[ttype]))
                box_labels.append(f'${ttype}_{{J}}$')
            if tx_agg_ot[ttype]:
                box_data.append(pd.Series(tx_agg_ot[ttype]))
                box_labels.append(f'${ttype}_{{O}}$')
        # Plot boxplot if data exists
        if box_data:
            bp = ax.boxplot(box_data, patch_artist=True, medianprops=dict(color="grey", linewidth=1.5),
                            showmeans=True,
                            meanprops={"marker": "x", "markerfacecolor": "red", "markeredgecolor": "red", "markersize": 5})
            ax.set_xticklabels(box_labels, rotation=0, ha='right')
            for patch in bp['boxes']:
                patch.set_facecolor('lightgrey')
        ax.set_title(f'Load: {int(load_level) * 3} T/s')
        ax.set_xlabel('')
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    # Set y-label for the first subplot
    axes[0].set_ylabel('Power (Watts)')
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved combined boxplot to {output_path}")


def main():
    """
    Main function to collect data and generate plots.
    """    
    trim_seconds = 60  # Number of seconds to trim at start and end of each time series
    # List of scenario suffixes to include and order
    scenario_suffixes = [
        #"docker_tools",
        #"idle_no_tools",
        #"docker_none",
        #"docker_idle",
        #"docker_kepler",
        #"docker_scaphandre",
        "docker_otjae",
        "docker_joularjx",
        "joularjx_tx",
        "otjae_tx"
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
        "docker_joularjx": "",
        "joularjx_tx": "JoularJX",
        "otjae_tx": "OTJAE"
    }
    # Specify which load levels to include in the evaluation (as strings)
    #included_load_levels = ["0", "230", "350", "480", "560"]  # Change as needed
    included_load_levels = ["230", "350", "480", "560"]
    data_by_load = collect_data_by_load_level(trim_seconds=trim_seconds, scenario_suffixes=scenario_suffixes, included_load_levels=included_load_levels)
    plot_all_load_levels_in_one_row(data_by_load, "./transaction_power_consumption_boxplots_all_loads.pdf", scenario_suffixes=scenario_suffixes, custom_labels=custom_labels)

    # Print mean values for each transaction and scenario

    print("\nMean power values per transaction and scenario:")
    for load_level, scenario_dict in data_by_load.items():
        if not str(load_level).isdigit():
            continue
        print(f"\nLoad level: {load_level}")
        for scenario, results in scenario_dict.items():
            # JoularJX transaction-level
            if 'joularjx_tx' in results and results['joularjx_tx']:
                joularjx_tx = results['joularjx_tx']
                steady_state_time = joularjx_tx.get('steady_state_time', None)
                total_energy_from_file = joularjx_tx.get('total_energy_from_file', {})
                if isinstance(joularjx_tx, dict) and 'per_invocation' in joularjx_tx and 'per_second' in joularjx_tx:
                    for ttype in ['GET', 'POST', 'DELETE']:
                        per_inv = joularjx_tx['per_invocation'].get(ttype, [])
                        per_sec = joularjx_tx['per_second'].get(ttype, [])
                        total_energy = joularjx_tx.get('total_energy', {}).get(ttype, None)
                        file_energy = total_energy_from_file.get(ttype, None)
                        if per_inv:
                            mean_inv = float(np.mean(per_inv))
                            print(f"  {scenario} - {ttype}_J: mean per-invocation = {mean_inv:.3f} W", end='')
                        if per_sec:
                            mean_sec = float(np.mean(per_sec))
                            if per_inv:
                                print(f", mean per-second = {mean_sec:.3f} W", end='')
                            else:
                                print(f"  {scenario} - {ttype}_J: mean per-second = {mean_sec:.3f} W", end='')
                        if total_energy is not None and steady_state_time and steady_state_time > 0:
                            mean_total = total_energy / steady_state_time
                            print(f", mean total-energy/steady-state = {mean_total:.3f} W", end='')
                        if file_energy is not None and steady_state_time and steady_state_time > 0:
                            mean_file = file_energy / steady_state_time
                            print(f", mean file-energy/steady-state = {mean_file:.3f} W")
                        elif per_inv or per_sec:
                            print()
                else:
                    # fallback: old structure
                    for ttype, values in joularjx_tx.items():
                        if values:
                            mean_val = float(np.mean(values))
                            print(f"  {scenario} - {ttype}_J: mean = {mean_val:.3f} W")
            # OTJAE transaction-level
            if 'otjae_tx' in results and results['otjae_tx']:
                otjae_tx = results['otjae_tx']
                if isinstance(otjae_tx, dict) and 'per_invocation' in otjae_tx and 'per_second' in otjae_tx:
                    for ttype in ['GET', 'POST', 'DELETE']:
                        per_inv = otjae_tx['per_invocation'].get(ttype, [])
                        per_sec = otjae_tx['per_second'].get(ttype, [])
                        if per_inv:
                            mean_inv = float(np.mean(per_inv))
                            print(f"  {scenario} - {ttype}_O: mean per-invocation = {mean_inv:.3f} W", end='')
                        if per_sec:
                            mean_sec = float(np.mean(per_sec))
                            if per_inv:
                                print(f", mean per-second = {mean_sec:.3f} W")
                            else:
                                print(f"  {scenario} - {ttype}_O: mean per-second = {mean_sec:.3f} W")
                        elif per_inv:
                            print()
                else:
                    # fallback: old structure
                    for ttype, values in otjae_tx.items():
                        if values:
                            mean_val = float(np.mean(values))
                            print(f"  {scenario} - {ttype}_O: mean = {mean_val:.3f} W")



    # Generate LaTeX table of mean transaction power values using per_second mean for both OTJAE and JoularJX, and total energy/steady-state time for JoularJX (both calculated and file-based)
    # Structure: {load_level: {ttype: {'OTJAE': mean, 'JoularJX': mean, 'JoularJX_total': mean, 'JoularJX_file_total': mean}}}
    table_means = {}
    for load_level, scenario_dict in data_by_load.items():
        if not str(load_level).isdigit():
            continue
        tx_means = {'GET': {'OTJAE': None, 'JoularJX': None, 'JoularJX_total': None, 'JoularJX_file_total': None},
                    'POST': {'OTJAE': None, 'JoularJX': None, 'JoularJX_total': None, 'JoularJX_file_total': None},
                    'DELETE': {'OTJAE': None, 'JoularJX': None, 'JoularJX_total': None, 'JoularJX_file_total': None}}
        agg_otjae_per_second = {'GET': [], 'POST': [], 'DELETE': []}
        agg_joularjx_per_second = {'GET': [], 'POST': [], 'DELETE': []}
        agg_joularjx_total_energy = {'GET': 0.0, 'POST': 0.0, 'DELETE': 0.0}
        agg_joularjx_file_total_energy = {'GET': [], 'POST': [], 'DELETE': []}  # Now a list of file energies per run
        agg_joularjx_total_time = []
        for scenario, results in scenario_dict.items():
            # JoularJX: use per_second and total_energy if new structure, else fallback to old
            if 'joularjx_tx' in results and results['joularjx_tx']:
                joularjx_tx = results['joularjx_tx']
                if isinstance(joularjx_tx, dict) and 'per_second' in joularjx_tx:
                    for ttype, values in joularjx_tx['per_second'].items():
                        if values:
                            agg_joularjx_per_second[ttype].extend(values)
                    # Sum total energy and time for each transaction type
                    total_energy = joularjx_tx.get('total_energy', {})
                    file_total_energy = joularjx_tx.get('total_energy_from_file', {})
                    steady_state_time = joularjx_tx.get('steady_state_time', None)
                    if steady_state_time and steady_state_time > 0:
                        agg_joularjx_total_time.append(steady_state_time)
                        for ttype in ['GET', 'POST', 'DELETE']:
                            agg_joularjx_total_energy[ttype] += total_energy.get(ttype, 0.0)
                            # Instead of summing, collect all file energies as a list
                            val = file_total_energy.get(ttype, None)
                            if val is not None:
                                agg_joularjx_file_total_energy[ttype].append(val)
                else:
                    for ttype, values in joularjx_tx.items():
                        if values:
                            agg_joularjx_per_second[ttype].extend(values)
            # OTJAE: use per_second if new structure, else fallback to old
            if 'otjae_tx' in results and results['otjae_tx']:
                otjae_tx = results['otjae_tx']
                if isinstance(otjae_tx, dict) and 'per_second' in otjae_tx:
                    for ttype, values in otjae_tx['per_second'].items():
                        if values:
                            agg_otjae_per_second[ttype].extend(values)
                else:
                    for ttype, values in otjae_tx.items():
                        if values:
                            agg_otjae_per_second[ttype].extend(values)
        # For JoularJX total energy, use the mean of all steady-state times (should be the same for all runs)
        mean_steady_state_time = np.mean(agg_joularjx_total_time) if agg_joularjx_total_time else None
        for ttype in ['GET', 'POST', 'DELETE']:
            if agg_otjae_per_second[ttype]:
                tx_means[ttype]['OTJAE'] = float(np.mean(agg_otjae_per_second[ttype]))
            if agg_joularjx_per_second[ttype]:
                tx_means[ttype]['JoularJX'] = float(np.mean(agg_joularjx_per_second[ttype]))
            # For JoularJX_file_total, divide by total experiment duration: steady_state_time + 2*trim_seconds
            if mean_steady_state_time and mean_steady_state_time > 0:
                tx_means[ttype]['JoularJX_total'] = agg_joularjx_total_energy[ttype] / mean_steady_state_time
                experiment_total_time = mean_steady_state_time + 2 * trim_seconds
            else:
                tx_means[ttype]['JoularJX_total'] = None
                experiment_total_time = None
            if agg_joularjx_file_total_energy[ttype] and experiment_total_time and experiment_total_time > 0:
                mean_file_energy = np.mean(agg_joularjx_file_total_energy[ttype])
                tx_means[ttype]['JoularJX_file_total'] = mean_file_energy / experiment_total_time
            else:
                tx_means[ttype]['JoularJX_file_total'] = None
        table_means[load_level] = tx_means


    print("\nLaTeX table of mean transaction power values (copy-paste into your document):\n")
    print(r"""\begin{table*} [h]
    \begin{center}
        \begin{tabular}{ |c|c|c|c|c|c|}
            \hline
            \multirow{2}{*}{\small{Load}} & \multirow{2}{*}{\small{Transaction}} & \multicolumn{2}{|c|}{OTJAE} & \multicolumn{2}{|c|}{JoularJX} \\
            \cline{3-6}
            &  & P\textsubscript{T}  &   P\textsubscript{$\Delta$}& P\textsubscript{T} & P\textsubscript{$\Delta$} \\
            \hline""")
    for load_level in sorted(table_means.keys(), key=lambda x: int(x)):
        tx_means = table_means[load_level]
        load_label = f"{int(load_level)}T/s"
        for idx, ttype in enumerate(['GET', 'POST', 'DELETE']):
            otjae_val = tx_means[ttype]['OTJAE']
            joularjx_file_total_val = tx_means[ttype]['JoularJX_file_total']
            otjae_str = f"{otjae_val:.2f}W" if otjae_val is not None else ""
            joularjx_file_total_str = f"{joularjx_file_total_val:.2f}W" if joularjx_file_total_val is not None else ""
            row_load = load_label if idx == 0 else load_label
            # Only fill Pt columns, leave Pdelta columns empty
            if idx == 0:
                print(f"            {row_load} & {ttype} & {otjae_str} &  \\multirow{{3}}{{*}}{{}} & {joularjx_file_total_str} &   \\multirow{{3}}{{*}}{{}} \\\\")
            else:
                print(f"            {row_load} & {ttype} & {otjae_str} &  & {joularjx_file_total_str} &   \\\\ ")
            if idx < 2:
                print("            \\cline{1-3}\\cline{5-5}")
        print("            \\hline\n            \\hline")
    print(r"""        \end{tabular}
                \caption{Mean transaction power consumption per second by load level}
                \label{tab:power_consumption_transaction}
            \end{center}
        \end{table*}""")


# Run the script if executed directly
if __name__ == "__main__":
    main()
