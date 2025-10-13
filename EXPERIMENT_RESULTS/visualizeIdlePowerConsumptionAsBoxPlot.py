
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def trim_time_series(df, trim_seconds):
    """
    Removes first and last records based on the specified time window
    """
    if trim_seconds <= 0:
        return df

    # Get the start and end timestamps
    start_time = df['datetime'].min()
    end_time = df['datetime'].max()

    # Calculate the cutoff times
    cutoff_start = start_time + pd.Timedelta(seconds=trim_seconds)
    cutoff_end = end_time - pd.Timedelta(seconds=trim_seconds)

    # Filter the dataframe
    return df[(df['datetime'] >= cutoff_start) & (df['datetime'] <= cutoff_end)]

def load_rittal_data(file_path, trim_seconds=0):
    """
    Loads Rittal data and sums up power values per timestamp
    """
    df = pd.read_csv(file_path)
    # Group by timestamp and sum power values
    power_data = df.groupby('Timestamp')['Power (Watts)'].sum().reset_index()
    power_data['datetime'] = pd.to_datetime(power_data['Timestamp'], unit='ms')

    # Trim the time series if requested
    if trim_seconds > 0:
        power_data = trim_time_series(power_data, trim_seconds)

    return power_data

def calculate_power_from_energy(df, trim_seconds=0):
    """
    Calculates power from energy values in powercap data
    """
    # Sort by timestamp and domain
    df = df.sort_values(['Timestamp', 'Domain'])

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['Timestamp'], unit='ms')

    # Calculate energy and time differences per domain
    power_data = []

    for domain in df['Domain'].unique():
        domain_data = df[df['Domain'] == domain].copy()

        # Calculate differences
        energy_diff = domain_data[' Energy (micro joules)'].diff()
        time_diff = domain_data['datetime'].diff().dt.total_seconds()

        # Calculate power in watts (micro joules to watts)
        power = energy_diff / (time_diff * 1_000_000)

        # Check for negative power values
        # This can happen when the Intel RAPL counters go over the maximum value and reset
        # e.g., in this scenario:
        # 1758011446001,package-0,262104537750,-1
        # 1758011447000,package-0,12592130,-1
        # In such a scenario we skip the current second and continue with the next and print a warning
        if (power < 0).any():
            negative_indices = power[power < 0].index
            for idx in negative_indices:
                print(f"\nNegative power value found for Domain {domain}:")
                print(f"Timestamp: {domain_data.loc[idx, 'datetime']} {domain_data.loc[idx, 'Timestamp']}")
                print(f"Energy diff: {energy_diff[idx]} µJ")
                print(f"Time diff: {time_diff[idx]} s")
                print(f"Calculated power: {power[idx]} W")
                print("\nInvolved records:")
                print(f"Current record:")
                print(domain_data.loc[idx, [' Energy (micro joules)', 'datetime']].to_string())
                #print(f"Vorheriger Datensatz:")
                #print(domain_data.loc[idx-1, [' Energy (micro joules)', 'datetime']].to_string())
                #raise ValueError(f"Negative Leistung ({power[idx]:.2f} W) berechnet für Domain {domain}")
                print(f"Skipping negative power value for Domain {domain} as RAPL might have overflown")
                power[idx] = np.nan  # Setze negative Werte auf NaN

        domain_data['Power'] = power
        domain_data = domain_data.dropna(subset=['Power'])
        power_data.append(domain_data)

    # Kombiniere Domains und summiere die Leistung nur für Zeitstempel,
    # die in allen Domains vorhanden sind
    result = pd.concat(power_data)

    # Gruppiere nach Zeitstempel und zähle die Anzahl der Domains
    domain_counts = result.groupby('datetime')['Domain'].count()

    # Filtere nur die Zeitstempel, die Daten von allen Domains haben
    complete_timestamps = domain_counts[domain_counts == len(df['Domain'].unique())].index

    # Filtere die Ergebnisse auf diese Zeitstempel und summiere die Leistung
    result_filtered = result[result['datetime'].isin(complete_timestamps)]
    power_sum = result_filtered.groupby('datetime')['Power'].sum().reset_index()

    # Trim the time series if requested
    if trim_seconds > 0:
        power_sum = trim_time_series(power_sum, trim_seconds)

    return power_sum

def load_power_data(file_path, trim_seconds=0):
    """
    Loads energy data based on file type and converts to power data
    """
    if 'rittal' in file_path.lower():
        return load_rittal_data(file_path, trim_seconds)
    elif 'powercap' in file_path.lower():
        df = pd.read_csv(file_path)
        return calculate_power_from_energy(df, trim_seconds)
    else:
        raise ValueError(f"Unknown file type: {file_path}")


def create_power_consumption_boxplot(data_dirs, output_path, custom_labels=None, trim_seconds=0):
    """
    Creates a boxplot of power consumption for different experiments with separate y-axes
    for RAPL and EM measurements
    Optionally, you can specify 'scenario_order' (list of suffixes) to control the order of boxplots.
    """
    scenario_order = [
        "idle_no_tools",
        "docker_none",
        "docker_idle",
        "docker_kepler",
        "docker_scaphandre",
        "docker_otjae",
        "docker_joularjx"]  # Set to a list of suffixes to control order, or pass as argument
    # --- New logic: group by scenario suffix, aggregate all runs ---
    from collections import defaultdict
    scenario_rittal = defaultdict(list)
    scenario_powercap = defaultdict(list)
    scenario_labels = {}

    for dir_path in data_dirs:
        dir_path = Path(dir_path)
        # Extract scenario key: everything after the timestamp (4th underscore)
        # Example: 20250923_174052_baseline_idle_no_tools -> idle_no_tools
        parts = dir_path.name.split('_', 3)
        if len(parts) < 4:
            print(f"Directory name format unexpected: {dir_path.name}")
            continue
        scenario_key = parts[3]
        # Use custom label if available, else scenario_key
        label = custom_labels.get(scenario_key) if custom_labels else scenario_key
        scenario_labels[scenario_key] = label

        power_files = list(dir_path.glob('**/rittal_*.csv')) + list(dir_path.glob('**/powercap_*.csv'))
        if not power_files:
            print(f"No power data found in: {dir_path}")
            continue
        for file_path in power_files:
            print(f"Using data file: {file_path}")
            try:
                power_data = load_power_data(str(file_path), trim_seconds)
                if 'Power' in power_data.columns:
                    power_values = power_data['Power']
                else:
                    power_values = power_data['Power (Watts)']
                if 'rittal' in file_path.name.lower():
                    scenario_rittal[scenario_key].append(power_values)
                else:
                    scenario_powercap[scenario_key].append(power_values)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")


    # Determine the order of scenarios
    if scenario_order is not None:
        ordered_rittal = [(k, scenario_rittal[k]) for k in scenario_order if k in scenario_rittal]
        ordered_powercap = [(k, scenario_powercap[k]) for k in scenario_order if k in scenario_powercap]
    else:
        ordered_rittal = list(scenario_rittal.items())
        ordered_powercap = list(scenario_powercap.items())

    # Combine all runs for each scenario (only include scenarios with data)
    rittal_data = []
    rittal_labels = []
    for scenario_key, runs in ordered_rittal:
        if runs:
            combined = pd.concat(runs, ignore_index=True)
            rittal_data.append(combined)
            rittal_labels.append(scenario_labels.get(scenario_key, scenario_key))

    powercap_data = []
    powercap_labels = []
    for scenario_key, runs in ordered_powercap:
        if runs:
            combined = pd.concat(runs, ignore_index=True)
            powercap_data.append(combined)
            powercap_labels.append(scenario_labels.get(scenario_key, scenario_key))

    if not (rittal_data or powercap_data):
        print("No data found for plotting")
        return

    # Create figure with two y-axes and specify relative widths (1:3 ratio)
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0])  # EM plot (larger)
    ax2 = fig.add_subplot(gs[1])  # RAPL plot (smaller)


    # Plot Rittal data on the left subplot
    if rittal_data:
        bp2 = ax1.boxplot(rittal_data, patch_artist=True,medianprops=dict(color="grey", linewidth=1.5),
                          showmeans=True,
                          meanprops={"marker":"x",
                                     "markerfacecolor":"red",
                                     "markeredgecolor":"red",
                                     "markersize":5}
                          )
        ax1.set_xticklabels(rittal_labels, rotation=0, ha='center')
        ax1.tick_params(axis='x', pad=10)  # Erhöht den Abstand zwischen Plot und Labels
        ax1.set_ylabel('Power (Watts)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        # Add grid lines
        ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
        # Color the boxes
        for patch in bp2['boxes']:
            patch.set_facecolor('lightgrey')
        ax1.set_title('Idle Power Consumption $P_{EM}$')

    # Plot RAPL data on the right subplot
    if powercap_data:
        bp1 = ax2.boxplot(powercap_data, patch_artist=True,medianprops=dict(color="grey", linewidth=1.5),
                          showmeans=True,
                          meanprops={"marker":"x",
                                     "markerfacecolor":"white",
                                     "markeredgecolor":"red",
                                     "markersize":5}
                          )
        ax2.set_xticklabels(powercap_labels, rotation=0, ha='center')
        ax2.tick_params(axis='x', pad=10)  # Erhöht den Abstand zwischen Plot und Labels

        ax2.set_ylabel('Power (Watts)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        # Add grid lines
        ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
        # Color the boxes
        for patch in bp1['boxes']:
            patch.set_facecolor('lightgrey')
        ax2.set_title('Idle Power Consumption $P_{S}$')

    # plt.suptitle('Idle Power Consumption')
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    #plt.show()
    plt.close()

    # Print statistics
    print("\Statistics:")
    print("-" * 50)

    if rittal_data:
        print("\nEM (Rittal) Measurements:")
        print("-" * 20)
        for data, label in zip(rittal_data, rittal_labels):
            mean_value = np.mean(data)
            median_value = np.median(data)
            print(f"\n{label}:")
            print(f"Mean: {mean_value:.2f} Watt")
            print(f"Median: {median_value:.2f} Watt")

    if powercap_data:
        print("\nRAPL Measurements:")
        print("-" * 20)
        for data, label in zip(powercap_data, powercap_labels):
            mean_value = np.mean(data)
            median_value = np.median(data)
            print(f"\n{label}:")
            print(f"Mean: {mean_value:.2f} Watt")
            print(f"Median: {median_value:.2f} Watt")



# Automatically collect all run folders (./0/, ./0_run2/, ./0_run3/) and all scenario subfolders
import glob
base_run_folders = ["./EXPERIMENT_RESULTS/0", "./EXPERIMENT_RESULTS/0_run2", "./EXPERIMENT_RESULTS/0_run3"]
data_directories = []
for run_folder in base_run_folders:
    scenario_dirs = glob.glob(f"{run_folder}/*/")
    data_directories.extend(scenario_dirs)

# Optional: Custom labels for scenario suffixes (applies to all runs)
custom_labels_by_suffix = {
    "idle_no_tools": "Idle",
    "docker_none": "Container\nIdle (CI)",
    "docker_idle": "CI, Powercap (PC)\nand ProcFS",
    "docker_kepler": "CI, PC, ProcFS\nand Kepler",
    "docker_scaphandre": "CI, PC, ProcFS\nand Scaphandre",
    "docker_otjae": "CI, PC, ProcFS\nand OTJAE",
    "docker_joularjx": "CI, PC, ProcFS\nand JoularJX"
    # Add more custom labels here
}

create_power_consumption_boxplot(data_directories, "./idle_power_consumption_boxplot.pdf", custom_labels_by_suffix, trim_seconds=60) # Trimmt 60 Sekunden am Anfang und Ende

