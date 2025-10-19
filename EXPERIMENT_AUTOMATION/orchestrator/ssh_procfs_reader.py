# ssh_procfs_reader.py
"""
Run the remote ProcFS reader JAR via SSH, collect its CSV, clean warnings, and log metadata.

Behavior
--------
- Starts the reader on the remote host in the background and captures its PID.
- Chooses metric flags based on whether PIDs were provided (adds --io only when PIDs exist).
- Waits `duration` seconds, then stops the process (if still running).
- Downloads the produced CSV into {output_dir}/{YYYYMMDD}/procfs_{...}.csv
- Removes lines starting with "[WARN]" from the CSV (in-place by default).
- Deletes the remote CSV after a successful download.
- Logs a JSON line with run metadata to {output_dir}/logs/experiment_log.jsonl

Notes
-----
- Public function names, parameters, and printed messages are preserved.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import datetime
import json
import os
import time
from typing import Optional, List

import paramiko
from orchestrator.ssh_temperatures import get_remote_temperatures

# ──────────────────────────────────────────────────────────────────────────────
# Logging helper (unchanged signature)
# ──────────────────────────────────────────────────────────────────────────────

def log_experiment(metadata: dict, log_path: str) -> None:
    """
    Append a single-line JSON record to the given log file.
    Creates parent folders if missing.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(metadata) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_remote_procfs(
    hostname: str,
    username: str,
    password: str,
    remote_dir: str,
    procfs_jar_filename: str,
    pids: list,
    experiment_type: str,
    iteration: int,
    total_iterations: int,
    duration: int,
    interval: str = "*/1 * * * * ?",
    output_dir: str = "./output",
) -> None:
    """
    Execute the ProcFS reader remotely and fetch the resulting CSV.

    Parameters
    ----------
    hostname, username, password : str
        SSH credentials for the target host.
    remote_dir : str
        Remote working directory where the JAR resides and CSV is written.
    procfs_jar_filename : str
        ProcFS reader JAR filename as it exists on the remote host.
    pids : list[str]
        Target process IDs (strings). If empty, '--io' is omitted.
    experiment_type : str
        Label for filename tagging.
    iteration, total_iterations : int
        Iteration counters for filename tagging.
    duration : int
        Time in seconds to let the reader run before stopping it.
    interval : str
        CRON-like expression for the reader sampling interval.
    output_dir : str
        Local root output folder (default: ./output).

    Side Effects
    ------------
    - Writes CSV to local {output_dir}/{YYYYMMDD}/procfs_{...}.csv
    - Prints status lines; logs metadata JSON into {output_dir}/logs/experiment_log.jsonl
    """
    # ── Output naming & folders (aligned with Powercap) ───────────────────────
    output_dir = os.path.abspath(output_dir)
    date_tag = datetime.datetime.now().strftime("%Y%m%d")
    time_tag = datetime.datetime.now().strftime("%H%M%S")
    base_filename = f"procfs_{experiment_type}_{date_tag}_{time_tag}_{iteration}_{total_iterations}.csv"

    # Normalize remote paths
    remote_dir = remote_dir.rstrip("/")
    remote_output_file = f"{remote_dir}/{base_filename}"

    # Build PID arg string once (keep original prints/behavior)
    pid_args = " ".join(pids) if pids else ""

    # ── SSH connect ───────────────────────────────────────────────────────────
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username=username, password=password)

    try:
        # Prepare metric flags based on PID presence (preserve original behavior)
        metric_flags: List[str] = ["--cpu", "--network", "--memory"]
        if not pids:
            print("[~] No PIDs provided – skipping '--io' and PID args for procfs-reader.")
        else:
            metric_flags.append("--io")

        metrics = " ".join(metric_flags)

        # Start procfs-reader remotely (background) and echo PID
        java_cmd = (
            f"sudo sh -c 'cd {remote_dir} && "
            f"java -Dprocfs.cron=\"{interval}\" "
            f"-jar {procfs_jar_filename} {metrics} {pid_args} "
            f"> {base_filename} 2>&1 & echo $!'"
        )

        stdin, stdout, stderr = ssh.exec_command(java_cmd, get_pty=True)

        # Wait up to 5 seconds for remote PID
        procfs_pid = ""
        start_wait = time.time()
        while True:
            if stdout.channel.recv_ready():
                procfs_pid = stdout.readline().strip()
                break
            if time.time() - start_wait > 5:
                print(f"[!] Timeout while waiting for PID from remote command: {java_cmd}")
                break
            time.sleep(0.1)

        err_output = stderr.read().decode().strip()
        if err_output:
            print(f"[!] Error while starting remote procfs-reader: {err_output}")
        if not procfs_pid.isdigit():
            print(f"[✗] Invalid or missing remote PID: '{procfs_pid}'")
            return

        print(f"[~] Start remote procfs-reader (PID: {procfs_pid}) for PIDs: {pid_args}")
        temperature_before = get_remote_temperatures(ssh)
        start_time = datetime.datetime.now().isoformat()

        # Measurement window
        time.sleep(max(0, duration))

        # Kill reader process (best effort)
        check_cmd = f"ps -p {procfs_pid} > /dev/null && echo RUNNING || echo GONE"
        _, stdout, _ = ssh.exec_command(check_cmd)
        state = stdout.read().decode().strip()

        if state == "RUNNING":
            ssh.exec_command(f"sudo kill {procfs_pid}")
            print(f"[~] Stop remote procfs-reader (PID: {procfs_pid})")
        else:
            print(f"[~] Remote procfs-reader already exited (PID: {procfs_pid})")

        time.sleep(1)  # small flush window

        # ── SFTP fetch & cleanup ──────────────────────────────────────────────
        sftp: Optional[paramiko.SFTPClient] = None
        local_output_file = os.path.join(output_dir, base_filename)
        try:
            sftp = ssh.open_sftp()
            sftp.stat(remote_output_file)  # raise if missing

            os.makedirs(os.path.dirname(local_output_file), exist_ok=True)
            sftp.get(remote_output_file, local_output_file)
            print(f"[↓] Save remote procfs-reader to: {local_output_file}")

            # Clean the procfs-reader file in place
            clean_procfs_file(local_output_file, local_output_file)
            print(f"[~] Cleaned procfs-reader file: {local_output_file}")

            sftp.remove(remote_output_file)
            print(f"[~] Delete remote procfs-reader file: {remote_output_file}")

        except FileNotFoundError:
            print(f"[✗] Remote output file not found: {remote_output_file}")
            return
        except Exception as e:
            print(f"[✗] Error during procfs-reader execution: {e}")
            return
        finally:
            try:
                if sftp:
                    sftp.close()
            except Exception:
                pass

        temperature_after = get_remote_temperatures(ssh)
        end_time = datetime.datetime.now().isoformat()

    finally:
        ssh.close()

    # ── Post-run metadata logging (best effort) ───────────────────────────────
    try:
        output_size = os.path.getsize(local_output_file)
    except Exception:
        output_size = 0

    log_path = os.path.join(output_dir, "logs", "experiment_log.jsonl")
    log_experiment(
        {
            "experiment": experiment_type,
            "iteration": iteration,
            "total_iterations": total_iterations,
            "duration": duration,
            "pid": int(procfs_pid) if procfs_pid.isdigit() else None,
            "service_pids": pids,
            "start_time": start_time,  # more accurate than previous "approx end"
            "end_time": end_time,
            "temperature_before": temperature_before,
            "temperature_after": temperature_after,
            "output_file": base_filename,
            "output_size": output_size,
        },
        log_path,
    )


# ──────────────────────────────────────────────────────────────────────────────
# File cleaning helper
# ──────────────────────────────────────────────────────────────────────────────

def clean_procfs_file(input_path: str, output_path: Optional[str] = None) -> None:
    """
    Removes lines starting with '[WARN]' from a procfs CSV file.

    Parameters
    ----------
    input_path : str
        Path to the input procfs CSV file.
    output_path : str, optional
        Path to save the cleaned file. If None, overwrites the original.
    """
    output_path = output_path or input_path

    with open(input_path, "r", encoding="utf-8", errors="replace") as infile:
        lines = infile.readlines()

    # Filter out any lines starting with [WARN]
    clean_lines = [line for line in lines if not line.lstrip().startswith("[WARN]")]

    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.writelines(clean_lines)

    print(f"[✓] Cleaned file saved to: {output_path}")
