import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Konfiguration: Verzeichnisse mit neuen powercap-Dateien (Energie in µJ)
# Erwartetes Dateimuster: powercap_*.csv in den Last-Unterordnern
base_dir = Path("./")
last_to_load = {
    "230": "690",
    "350": "1050",
    "480": "1440",
    "560": "1680",
}
run_mapping = {
    "spring_docker_otjae": "OTJAE",
    "spring_docker_joularjx": "JoularJX",
    "spring_docker_scaphandre": "Scaphandre",
    "spring_docker_kepler": "Kepler",
    "spring_docker_tools": "none",  # falls benötigt
}

def find_powercap_files():
    rows = []
    for last_dir, load_label in last_to_load.items():
        dir_path = base_dir / last_dir
        if not dir_path.exists():
            continue
        for run_folder in dir_path.iterdir():
            if not run_folder.is_dir():
                continue
            # Extract numeric load level (handles _run2, _run3, etc.)
            folder_name = run_folder.name
            for key, val in run_mapping.items():
                if key in folder_name:
                    run_name = val
                    break
            else:
                run_name = None
            if run_name is None:
                continue
            # Numeric load extraction (e.g., 230, 230_run2, 230_run3 -> 230)
            numeric_load = last_dir
            for f in run_folder.glob("**/powercap_*.csv"):
                rows.append({"file": str(f), "load": load_label, "numeric_load": numeric_load, "run": run_name})
    return rows

def energy_to_power(df: pd.DataFrame) -> pd.DataFrame:
    # Spalten normalisieren
    # Erwartete Spalten: Timestamp, Domain, Energy (micro joules), DRAM Energy (micro joules)
    # Timestamp in ms (int), Energien kumuliert in µJ
    df = df.rename(columns={
        " Energy (micro joules)": "Energy (micro joules)",
        " DRAM Energy (micro joules)": "DRAM Energy (micro joules)",
    })
    # String-Trim für evtl. führende/trailing Spaces
    df.columns = [c.strip() for c in df.columns]

    # Nur Pakete (Domain beginnt mit 'package-')
    df = df[df["Domain"].astype(str).str.startswith("package-")].copy()

    # Nach Domain und Timestamp sortieren
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce").astype("int64")
    df = df.sort_values(["Domain", "Timestamp"])

    # Differenzen je Domain berechnen
    for col in ["Energy (micro joules)", "DRAM Energy (micro joules)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["dE_pkg_uJ"] = df.groupby("Domain")["Energy (micro joules)"].diff()
    # DRAM ist ggf. -1 -> ignorieren
    if "DRAM Energy (micro joules)" in df.columns:
        dram_valid = df["DRAM Energy (micro joules)"].where(df["DRAM Energy (micro joules)"] >= 0, np.nan)
        df["dE_dram_uJ"] = df.groupby("Domain")[dram_valid.name].diff()
    else:
        df["dE_dram_uJ"] = np.nan

    df["dt_ms"] = df.groupby("Domain")["Timestamp"].diff()

    # Leistung berechnen: P = dE/dt; µJ/ms == mW
    # In Watt: (µJ/ms)/1000
    df["P_pkg_W"] = (df["dE_pkg_uJ"] / df["dt_ms"]) / 1000.0
    df["P_dram_W"] = (df["dE_dram_uJ"] / df["dt_ms"]) / 1000.0

    # Nur gültige Zeilen (positive dt und dE)
    df = df[(df["dt_ms"] > 0) & (df["dE_pkg_uJ"] >= 0)]

    # Über Domains aggregieren (Summe der Packages)
    agg = df.groupby("Timestamp").agg(
        P_pkg_W=("P_pkg_W", "sum"),
        P_dram_W=("P_dram_W", "sum"),
    ).reset_index()

    # Totalleistung
    agg["Total Power (Watts)"] = agg["P_pkg_W"].fillna(0) + agg["P_dram_W"].fillna(0)

    return agg

def load_all_runs(file_rows):
    data_frames = []
    for info in file_rows:
        df_raw = pd.read_csv(info["file"])
        df_pow = energy_to_power(df_raw)
        t0 = df_pow["Timestamp"].min()
        df_pow["Relative Timestamp"] = df_pow["Timestamp"] - t0
        df_pow = df_pow[(df_pow["Relative Timestamp"] >= 60000) & (df_pow["Relative Timestamp"] <= 660000)].copy()
        # Use numeric_load for aggregation
        df_pow["load"] = info["numeric_load"]
        df_pow["run"] = info["run"]
        data_frames.append(df_pow)
    if not data_frames:
        return pd.DataFrame(columns=["Timestamp", "Total Power (Watts)", "load", "run"])
    return pd.concat(data_frames, ignore_index=True)

def main():
    csv_files = find_powercap_files()
    if not csv_files:
        print("Keine powercap_*.csv Dateien gefunden.")
        return

    df_all = load_all_runs(csv_files)

    # Boxplot: Total Power nach load/run
    load_order = ["230", "350", "480", "560"]
    # Farbpalette pro run (da zusätzliche Kategorien)
    run_order = ["none", "OTJAE", "JoularJX", "Scaphandre", "Kepler"]
    palette = sns.color_palette("Set2", n_colors=len(run_order))

    fig = plt.figure(figsize=(14, 10))
    ax = sns.boxplot(
        x="load",
        y="Total Power (Watts)",
        hue="run",
        data=df_all,
        order=load_order,
        hue_order=run_order,
        palette=palette,
        dodge=True
    )
    ax.yaxis.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc="upper left", title=None, fontsize=14)

    # Set x-tick labels to load*3
    new_labels = [str(int(l)*3) for l in load_order]
    ax.set_xticklabels(new_labels, fontsize=14)

    plt.xlabel("Load (T/s)", fontsize=14)
    plt.ylabel("$P_{S}$ (W)", fontsize=14)

    pdf_path = "../boxplot_total_power_by_load_and_run.pdf"
    plt.savefig(pdf_path, format="pdf")
    plt.show()
    print(f"Box plot successfully saved as PDF: {pdf_path}")

if __name__ == "__main__":
    main()