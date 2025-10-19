# ssh_powercap_reader.py
"""
Run the remote Powercap reader JAR via SSH, collect its CSV, and log metadata.

Behavior
--------
- Starts the reader on the remote host in the background and captures its PID.
- Waits `duration` seconds, then stops the process (if still running).
- Downloads the produced CSV into {output_dir}/{YYYYMMDD}/powercap_{...}.csv
- Deletes the remote CSV after a successful download.
- Logs a JSON line with run metadata to {output_dir}/logs/experiment_log.jsonl

Notes
-----
- Public function name, parameters, and printed messages are preserved.
- Uses `get_remote_temperatures(ssh)` before and after the run (best-effort).
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

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

def run_remote_powercap(
    hostname: str,
    username: str,
    password: str,
    jar_filename: str,
    remote_dir: str,
    duration: int,
    experiment_type: str,
    iteration: int,
    total_iterations: int,
    output_dir: str = "./output",
    interval: str = "*/1 * * * * ?",
) -> None:
    """
    Execute the Powercap reader remotely and fetch the resulting CSV.

    Parameters
    ----------
    hostname, username, password : str
        SSH credentials for the target host.
    jar_filename : str
        Powercap reader JAR filename as it exists on the remote host.
    remote_dir : str
        Remote working directory where the JAR resides and CSV is written.
    duration : int
        Time in seconds to let the reader run before stopping it.
    experiment_type : str
        Label for filename tagging.
    iteration, total_iterations : int
        Iteration counters for filename tagging.
    output_dir : str
        Local root output folder (default: ./output).
    interval : str
        CRON-like expression for the reader sampling interval.

    Side Effects
    ------------
    - Writes CSV to local {output_dir}/{YYYYMMDD}/powercap_{...}.csv
    - Prints status lines; logs metadata JSON into {output_dir}/logs/experiment_log.jsonl
    """
    # ── Naming & paths ────────────────────────────────────────────────────────
    output_dir = os.path.abspath(output_dir)
    date_tag = datetime.datetime.now().strftime("%Y%m%d")
    time_tag = datetime.datetime.now().strftime("%H%M%S")

    base_filename = f"powercap_{experiment_type}_{date_tag}_{time_tag}_{iteration}_{total_iterations}.csv"
    remote_output_file = f"{remote_dir.rstrip('/')}/{base_filename}"

    local_output_file = os.path.join(output_dir, base_filename)

    # ── SSH connect ───────────────────────────────────────────────────────────
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username=username, password=password)

    try:
        # Ensure JAR exists remotely
        remote_jar_path = f"{remote_dir.rstrip('/')}/{jar_filename}"
        _, stdout, _ = ssh.exec_command(f"test -f {remote_jar_path} && echo OK || echo MISSING")
        result = stdout.read().decode().strip()
        if result != "OK":
            print(f"[✗] Required remote powercap-reader JAR not found at {remote_jar_path} on {hostname}")
            return

        # Build command: run in background and echo PID
        # Use sudo sh -c so we can redirect output and get $! inside the same shell.
        java_cmd = (
            f"cd {remote_dir} && "
            f"sudo sh -c 'java -Dpowercap.cron=\"{interval}\" "
            f"-jar {jar_filename} > {base_filename} 2>&1 & echo $!'"
        )

        # Request a PTY to mimic interactive shell behavior (keeps prior behavior)
        _, stdout, stderr = ssh.exec_command(java_cmd, get_pty=True)

        # Wait for up to 5 seconds for PID to appear on stdout
        pid = ""
        start_wait = time.time()
        while True:
            if stdout.channel.recv_ready():
                pid_line = stdout.readline().strip()
                if pid_line:
                    pid = pid_line
                    break
            if time.time() - start_wait > 5:
                print(f"[!] Timeout while waiting for PID from remote command: {java_cmd}")
                break
            time.sleep(0.1)

        err = stderr.read().decode().strip()
        if err:
            print(f"[!] Error while starting remote powercap-reader: {err}")
        if not pid.isdigit():
            print(f"[✗] Invalid or missing remote PID: '{pid}'")
            return

        print(f"[~] Start remote powercap-reader (PID: {pid})")
        temperature_before = get_remote_temperatures(ssh)
        start_time = datetime.datetime.now().isoformat()

        # ── Run window ────────────────────────────────────────────────────────
        time.sleep(max(0, duration))

        # ── Stop remote process (best effort) ─────────────────────────────────
        check_cmd = f"ps -p {pid} > /dev/null && echo RUNNING || echo GONE"
        _, stdout, _ = ssh.exec_command(check_cmd)
        state = stdout.read().decode().strip()
        if state == "RUNNING":
            ssh.exec_command(f"sudo kill {pid}")
            print(f"[~] Stop remote powercap-reader (PID: {pid})")
        else:
            print(f"[~] Remote process {pid} already exited before kill.")

        # Give the process a moment to flush output
        time.sleep(2)

        # ── SFTP fetch & cleanup ──────────────────────────────────────────────
        sftp: Optional[paramiko.SFTPClient] = None
        try:
            sftp = ssh.open_sftp()
            # Ensure file exists before attempting download
            sftp.stat(remote_output_file)

            sftp.get(remote_output_file, local_output_file)
            print(f"[↓] Save remote powercap-reader to: {local_output_file}")

            # Remove remote file after successful download
            sftp.remove(remote_output_file)
            print(f"[~] Delete remote powercap-reader file: {remote_output_file}")

        except FileNotFoundError:
            print(f"[✗] Remote output file not found: {remote_output_file}")
            return
        except Exception as e:
            print(f"[✗] Unexpected error during download or cleanup: {e}")
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
            "pid": int(pid) if str(pid).isdigit() else None,
            "start_time": start_time,
            "end_time": end_time,
            "temperature_before": temperature_before,
            "temperature_after": temperature_after,
            "output_file": base_filename,
            "output_size": output_size,
        },
        log_path,
    )
