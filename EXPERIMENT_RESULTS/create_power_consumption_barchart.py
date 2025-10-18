"""
create_power_consumption_barchart.py

Utilities to load power measurement files (Rittal EM and Intel RAPL/powercap), aggregate them per load level and tool, and produce a combined bar chart comparing measurement types and scenario constants.

This module contains file loading helpers and a single convenience function `create_power_consumption_barchart` that walks experiment directories, reads Rittal and powercap CSVs, computes means and plots grouped bar charts.

This file is intended for offline analysis of experiment folders arranged under numeric load-level directories (e.g. `./230/`). It expects CSVs with certain column names as produced by the measurement tooling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def trim_time_series(df, trim_seconds):
    """
    Removes the first and last records based on the specified time window
    """
    if trim_seconds <= 0:
        return df

    start_time = df['datetime'].min()
    end_time = df['datetime'].max()

    cutoff_start = start_time + pd.Timedelta(seconds=trim_seconds)
    cutoff_end = end_time - pd.Timedelta(seconds=trim_seconds)

    return df[(df['datetime'] >= cutoff_start) & (df['datetime'] <= cutoff_end)]

def load_rittal_data(file_path, trim_seconds=0):
    """
    Loads Rittal data and sums power values per timestamp
    """
    df = pd.read_csv(file_path)
    power_data = df.groupby('Timestamp')['Power (Watts)'].sum().reset_index()
    power_data['datetime'] = pd.to_datetime(power_data['Timestamp'], unit='ms')

    if trim_seconds > 0:
        power_data = trim_time_series(power_data, trim_seconds)

    return power_data

def calculate_power_from_energy(df, trim_seconds=0):
    """
    Calculates power from energy values in Powercap data
    """
    df = df.sort_values(['Timestamp', 'Domain'])
    df['datetime'] = pd.to_datetime(df['Timestamp'], unit='ms')

    power_data = []

    for domain in df['Domain'].unique():
        domain_data = df[df['Domain'] == domain].copy()
        energy_diff = domain_data[' Energy (micro joules)'].diff()
        time_diff = domain_data['datetime'].diff().dt.total_seconds()
        power = energy_diff / (time_diff * 1_000_000)

        if (power < 0).any():
            power[power < 0] = np.nan

        domain_data['Power'] = power
        domain_data = domain_data.dropna(subset=['Power'])
        power_data.append(domain_data)

    result = pd.concat(power_data)
    domain_counts = result.groupby('datetime')['Domain'].count()
    complete_timestamps = domain_counts[domain_counts == len(df['Domain'].unique())].index
    result_filtered = result[result['datetime'].isin(complete_timestamps)]
    power_sum = result_filtered.groupby('datetime')['Power'].sum().reset_index()

    if trim_seconds > 0:
        power_sum = trim_time_series(power_sum, trim_seconds)

    return power_sum

def load_power_data(file_path, trim_seconds=0):
    """
    Loads energy data based on file type and converts it to power data
    """
    if 'rittal' in file_path.lower():
        return load_rittal_data(file_path, trim_seconds)
    elif 'powercap' in file_path.lower():
        df = pd.read_csv(file_path)
        return calculate_power_from_energy(df, trim_seconds)
    else:
        raise ValueError(f"Unknown file type: {file_path}")

def create_power_consumption_barchart(load_levels, output_path, trim_seconds=0):
    """
    Creates a combined bar chart of power consumption for different load levels,
    with RAPL and Rittal data side by side.
    Only considers files from directories ending with '_spring_docker_tools'.
    """
    # Constants for different scenarios (in Watts)
    SCENARIO_CONSTANTS = {
        'Kepler (Container/Process)': {
            0: 0,
            230: 56.56,
            350: 104.84,
            480: 186.55,
            560: 221.82
        },
        'Scaphandre (Container/Process)': {
            0: 0,
            230: 49.08,
            350: 108.98,
            480: 188.99,
            560: 229.18
        },
        'OTJAE (Process)': {
            0: 0,
            230: 142.06,
            350: 176.91,
            480: 208.50,
            560: 238.35
        },
        'JoularJX (Process)': {
            0: 0,
            230: 189.50,
            350: 213.05,
            480: 233.88,
            560: 244.16
        },
        'OTJAE (Transaction)': {
            0: 0,
            230: 138.44, #19.31+35.67+83.46
            350: 171.30, #23.47+44.48+103.35
            480: 200.92, #28.11+52.29+120.52
            560: 228.48, #32.69+59.29+136.50
        },
        'JoularJX (Transaction)': {
            0: 0,
            230: 182.91, #23.64+45.73+113.54
            350: 199.92,#24.45+50.05+125.42
            480: 219.03,#29.99+56.04+133.00
            560: 242.83,#49.07+66.48+127.28
        }
    }

    data = {
        'Load': [],
        'Type': [],
        'Tool': [],
        'Power': []
    }

    for load_level in load_levels:
        load_dir = Path(str(load_level))
        if not load_dir.exists():
            print(f"Directory not found: {load_dir}")
            continue

        # Search for all experiment directories ending with '_spring_docker_tools'
        experiment_dirs = [d for d in load_dir.glob("**/2025*") if d.is_dir() and d.name.endswith('_spring_docker_tools')]

        if not experiment_dirs:
            print(f"No directories ending with '_spring_docker_tools' found in: {load_dir}")
            continue

        for exp_dir in experiment_dirs:
            # Extract tool name from directory name
            tool_name = exp_dir.name.split('_')[-1]

            # Search for Rittal and Powercap files
            rittal_files = list(exp_dir.glob('**/rittal_*.csv'))
            powercap_files = list(exp_dir.glob('**/powercap_*.csv'))

            for r_file in rittal_files:
                try:
                    power_data = load_power_data(str(r_file), trim_seconds)
                    mean_power = power_data['Power (Watts)'].mean()
                    data['Load'].append(load_level)
                    data['Type'].append('Rittal')
                    data['Tool'].append(tool_name)
                    data['Power'].append(mean_power)
                except Exception as e:
                    print(f"Error loading {r_file}: {e}")

            for p_file in powercap_files:
                try:
                    power_data = load_power_data(str(p_file), trim_seconds)
                    mean_power = power_data['Power'].mean()
                    data['Load'].append(load_level)
                    data['Type'].append('RAPL')
                    data['Tool'].append(tool_name)
                    data['Power'].append(mean_power)
                except Exception as e:
                    print(f"Error loading {p_file}: {e}")

    # Add constant scenario data
    for scenario, load_values in SCENARIO_CONSTANTS.items():
        for load_level in load_levels:
            if load_level in load_values:
                data['Load'].append(load_level)
                data['Type'].append('Scenario')
                data['Tool'].append(scenario)
                data['Power'].append(load_values[load_level])

    # Create DataFrame from collected data
    df = pd.DataFrame(data)

    if df.empty:
        print("Warning: No data collected. Please check if the directories and files exist.")
        return

    # --- Color palette and tool mapping (from visualizePowerCapAsBoxplot.py) ---
    run_order = ["none", "OTJAE", "JoularJX", "Scaphandre", "Kepler"]
    tool_palette = dict(zip(run_order, sns.color_palette("Set2", n_colors=len(run_order))))
    em_color = "#888888"  # New color for EM (Rittal)

    fig, ax = plt.subplots(figsize=(18, 10))
    measured_tools = sorted([t for t in df['Tool'].unique() if t not in SCENARIO_CONSTANTS.keys()])
    scenario_tools = sorted([t for t in df['Tool'].unique() if t in SCENARIO_CONSTANTS.keys()])
    all_tools = measured_tools + scenario_tools
    x = np.arange(len(load_levels)) * 3.5
    total_width = 3.3
    total_bars = len(measured_tools) * 2 + len(scenario_tools)
    bar_width = total_width / total_bars
    bar_position = 0

    # Plot bars for measured tools
    for tool in measured_tools:
        tool_data = df[df['Tool'] == tool]
        # Rittal data (EM)
        rittal_data = tool_data[tool_data['Type'] == 'Rittal']
        rittal_means = [rittal_data[rittal_data['Load'] == load]['Power'].mean() for load in load_levels]
        pos_rittal = x + bar_position * bar_width - total_width/2
        bars_rittal = ax.bar(pos_rittal, rittal_means, bar_width,
                             label=f'EM',
                             color=em_color,
                             hatch='',
                             edgecolor='black',
                             linewidth=0.5)
        for i, (bar, value) in enumerate(zip(bars_rittal, rittal_means)):
            if not np.isnan(value) and value > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        '100%', ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=90)
        bar_position += 1
        # RAPL data
        rapl_data = tool_data[tool_data['Type'] == 'RAPL']
        rapl_means = [rapl_data[rapl_data['Load'] == load]['Power'].mean() for load in load_levels]
        pos_rapl = x + bar_position * bar_width - total_width/2
        # Map tool to run_order for color
        color = tool_palette.get(tool, '#333333')
        bars_rapl = ax.bar(pos_rapl, rapl_means, bar_width,
                           label=f'RAPL',
                           color=color,
                           hatch='////',
                           edgecolor='black',
                           linewidth=0.5)
        for i, (bar, rapl_val, rittal_val) in enumerate(zip(bars_rapl, rapl_means, rittal_means)):
            if not np.isnan(rapl_val) and not np.isnan(rittal_val) and rittal_val > 0:
                percentage = (rapl_val / rittal_val) * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=90)
        bar_position += 1

    # Scenario hatches for (Process)/(Transaction) distinction
    scenario_hatches = {'Process': '', 'Transaction': '///'}
    # Plot bars for scenario constants
    for idx, scenario in enumerate(scenario_tools):
        scenario_data = df[df['Tool'] == scenario]
        scenario_means = [scenario_data[scenario_data['Load'] == load]['Power'].mean() for load in load_levels]
        pos_scenario = x + bar_position * bar_width - total_width/2
        # Determine base tool and type
        base_tool = None
        scenario_type = ''
        for t in run_order:
            if scenario.lower().startswith(t.lower()):
                base_tool = t
                break
        if '(Transaction)' in scenario:
            scenario_type = 'Transaction'
        elif '(Process)' in scenario:
            scenario_type = 'Process'
        color = tool_palette.get(base_tool, '#333333')
        hatch_pattern = scenario_hatches.get(scenario_type, '')
        bars_scenario = ax.bar(pos_scenario, scenario_means, bar_width,
                               label=f'{scenario}',
                               color=color,
                               edgecolor='black',
                               linewidth=1.5,
                               hatch=hatch_pattern)
        for i, (bar, scenario_val) in enumerate(zip(bars_scenario, scenario_means)):
            if not np.isnan(scenario_val) and scenario_val > 0:
                load = load_levels[i]
                rittal_values = df[(df['Load'] == load) & (df['Type'] == 'Rittal')]['Power']
                if not rittal_values.empty:
                    rittal_mean = rittal_values.mean()
                    if rittal_mean > 0:
                        percentage = (scenario_val / rittal_mean) * 100
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=90)
        bar_position += 1

    # Chart labeling and formatting
    ax.set_xlabel('Load (T/s)', fontsize=16)
    ax.set_ylabel('Power (Watts)', fontsize=16)
    #ax.set_title('Power Consumption by Tool and Measurement Method', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(l)*3) for l in load_levels], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(loc='upper left', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add extra space at the top for percentage labels
    y_max = ax.get_ylim()[1]
    ax.set_ylim(top=y_max * 1.08)

    # Print statistics
    print("\nStatistics:")
    for load in load_levels:
        print(f"\nLoad level: {load} Requests/Second")
        load_data = df[df['Load'] == load]

        for tool in measured_tools:
            tool_data = load_data[load_data['Tool'] == tool]
            if not tool_data.empty:
                print(f"\n{tool}:")
                for measurement_type in ['Rittal', 'RAPL']:
                    type_data = tool_data[tool_data['Type'] == measurement_type]
                    if not type_data.empty:
                        mean_power = type_data['Power'].mean()
                        print(f"  {measurement_type}: {mean_power:.2f} Watts")

        # Print scenario constants
        for scenario in scenario_tools:
            scenario_data = load_data[load_data['Tool'] == scenario]
            if not scenario_data.empty:
                mean_power = scenario_data['Power'].mean()
                print(f"\n{scenario.upper()} (Scenario): {mean_power:.2f} Watts")

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

    # --- Print differences between transaction and process values for JOULARJX and OTJAE ---
    print("\nDifferences (Transaction - Process):")
    for tool in ["joularjx", "otjae"]:
        process_key = f"{tool} (Process)"
        transaction_key = f"{tool} (Transaction)"
        tool_upper = tool.upper()
        print(f"\n{tool_upper}:")
        for load in load_levels:
            process_val = SCENARIO_CONSTANTS.get(process_key, {}).get(load, None)
            transaction_val = SCENARIO_CONSTANTS.get(transaction_key, {}).get(load, None)
            if process_val is not None and transaction_val is not None:
                diff = transaction_val - process_val
                print(f"  Load {load}: {tool_upper} (Transaction) {transaction_val:.2f} - {tool_upper} (Process) {process_val:.2f} = {diff:.2f} W")
            else:
                print(f"  Load {load}: data missing")

# Example usage:
load_levels = [0, 230, 350, 480, 560]
create_power_consumption_barchart(load_levels,
                                  "../power_consumption_combined_barchart.pdf",
                                  trim_seconds=60)