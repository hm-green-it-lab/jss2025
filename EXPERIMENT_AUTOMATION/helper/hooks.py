# hooks.py
"""
Helpers to start/stop TeaStore in two modes:
- Docker Compose
- Individual Tomcat services (optionally instrumented)

Notes
-----
- Public function signatures are unchanged.
- Prints and general behavior remain the same.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports & Constants
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import stat
import time
from datetime import datetime
from typing import Dict, Tuple
import tarfile

import paramiko

# Order matters: registry must start first.
TOMCAT_SERVICES = [
    "tomcat-registry",      # Must be first entry!
    "tomcat-persistence",
    "tomcat-auth",
    "tomcat-recommender",
    "tomcat-image",
    "tomcat-webui",
]

# Tunables (kept as simple constants for readability, not exposed as API changes)
_DB_WARMUP_SECONDS = 5
_REGISTRY_WARMUP_SECONDS = 10
_SERVICE_START_DELAY_SECONDS = 5


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers (no API changes)
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_time_tag(exp: dict, config: dict) -> str:
    """
    Prefer a run-wide start tag if provided by the runner or set earlier.
    Fallback to now() only if nothing else exists.
    """
    return (
        exp.get("time_tag")
        or config.get("__time_tag__")
        or os.environ.get("EXPERIMENT_TIME_TAG")
        or datetime.now().strftime("%H%M%S")
    )

def _connect(hostname: str, username: str, password: str) -> paramiko.SSHClient:
    """
    Open an SSH connection to a host with password auth.

    Returns
    -------
    paramiko.SSHClient
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, username=username, password=password)
    return client


def _detect_services_root(client: paramiko.SSHClient, base_dir: str) -> str:
    """
    Detect the root folder containing the TeaStore Tomcat services.

    Supports both layouts:
      - <base_dir>/teastore/<service>
      - <base_dir>/<service>

    Returns
    -------
    str
        The path that contains the service folders.
    """
    candidate1 = f"{base_dir}/teastore"
    candidate2 = f"{base_dir}"
    for candidate in (candidate1, candidate2):
        test_cmd = f"[ -d {candidate}/tomcat-registry ] && echo OK || echo NO"
        _, stdout, _ = client.exec_command(test_cmd)
        if stdout.read().decode().strip() == "OK":
            return candidate
    # Fallback to the first candidate if checks fail (preserves previous behavior)
    return candidate1


def _extract_javaagent_fragment(opts: str) -> str:
    """
    Extract the most specific fragment for agent detection in /proc/<pid>/cmdline.

    Prefer the exact token starting with '-javaagent:' if present; otherwise use
    the generic '-javaagent:' marker.

    Returns
    -------
    str
        Fragment expected to appear in the JVM command line, or empty string.
    """
    if not opts:
        return ""
    for token in opts.split():
        if token.startswith("-javaagent:"):
            # e.g. -javaagent:/home/.../joularjx-3.0.1.jar
            return token
    return "-javaagent:"


def _check_agent_attached(client: paramiko.SSHClient, pid: str, expected_fragment: str) -> bool:
    """
    Verify that a JVM process contains the expected agent fragment.

    Strategy:
      1) Read /proc/<pid>/cmdline (NUL-separated); make it readable via `tr`.
      2) Fallback to `jcmd <pid> VM.flags` if cmdline is empty (best effort).

    Parameters
    ----------
    pid : str
        Target JVM PID (must be numeric).
    expected_fragment : str
        Substring to search for (e.g., '-javaagent:/path/to.jar').

    Returns
    -------
    bool
        True if the fragment appears in the process args, False otherwise.
    """
    if not pid.isdigit() or not expected_fragment:
        return False

    cmd = f"cat /proc/{pid}/cmdline | tr '\\0' ' '"
    _, stdout, _ = client.exec_command(cmd)
    cmdline = stdout.read().decode(errors="replace").strip()

    if not cmdline:
        # Best-effort fallback (jcmd may not be installed)
        _, stdout2, _ = client.exec_command(f"jcmd {pid} VM.flags 2>/dev/null | tr '\\0' ' '")
        cmdline = stdout2.read().decode(errors="replace").strip()

    return expected_fragment in cmdline


def _get_daily_out_dir(exp: dict) -> str:
    """
    Resolve the local daily output directory (e.g., ./output/20250816).

    If `exp['date_tag']` is provided by the runner, use it; otherwise derive
    YYYYMMDD from local time. The directory is created if missing.

    Returns
    -------
    str
        Absolute (or relative) path to the daily output folder.
    """
    base = exp.get("local_output_directory", "./output")
    date_tag = exp.get("date_tag") or datetime.now().strftime("%Y%m%d")
    daily = os.path.join(base, date_tag)
    os.makedirs(daily, exist_ok=True)
    return daily


def _sftp_isdir(sftp: paramiko.SFTPClient, path: str) -> bool:
    """
    Check whether an SFTP path exists and is a directory.
    """
    try:
        st = sftp.stat(path)
        return stat.S_ISDIR(st.st_mode)
    except (IOError, FileNotFoundError, OSError):
        return False


def _sftp_download_dir(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: str) -> None:
    """
    Recursively download a remote folder.

    Overwrites existing files with the same names locally.

    Parameters
    ----------
    remote_dir : str
        Remote directory path to download.
    local_dir : str
        Local target directory path.
    """
    os.makedirs(local_dir, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        r_path = f"{remote_dir.rstrip('/')}/{entry.filename}"
        l_path = os.path.join(local_dir, entry.filename)
        if stat.S_ISDIR(entry.st_mode):
            _sftp_download_dir(sftp, r_path, l_path)
        else:
            sftp.get(r_path, l_path)


def _write_setenv(client: paramiko.SSHClient, setenv_path: str, java_opts: str, catalina_opts: str) -> None:
    """
    Write/replace bin/setenv.sh via a here-doc. If strings are empty, create a
    minimal file (baseline run). Appends to existing env via $JAVA_OPTS/$CATALINA_OPTS.

    Parameters
    ----------
    setenv_path : str
        Absolute path to the setenv.sh file in a Tomcat service.
    java_opts : str
        Extra global JVM options (JAVA_OPTS). Usually empty.
    catalina_opts : str
        Tomcat server options (CATALINA_OPTS): heap, agents, runtime flags.
    """
    lines = ["#!/bin/bash"]
    if java_opts:
        lines.append(f'export JAVA_OPTS="{java_opts} $JAVA_OPTS"')
    if catalina_opts:
        lines.append(f'export CATALINA_OPTS="{catalina_opts} $CATALINA_OPTS"')

    content = "\n".join(lines) + "\n"
    cmd = (
        f"cat > {setenv_path} <<'EOF'\n{content}EOF\n"
        f"chmod +x {setenv_path}"
    )
    client.exec_command(cmd)

# ──────────────────────────────────────────────────────────────────────────────
# Docker helpers
# ──────────────────────────────────────────────────────────────────────────────

def start_docker(hostname: str, username: str, password: str, config: dict, output_dir: str) -> None:
    """
    Start using Docker Compose.

    Reads the command from:
        config['experiment']['remote_docker_start']

    Raises
    ------
    RuntimeError
        If the remote command returns a non-zero exit code.
    """
    print("[~] Starting Docker ...")
    compose_up_cmd = config['experiment']['remote_docker_start']

    client = _connect(hostname, username, password)
    try:
        _, stdout, stderr = client.exec_command(compose_up_cmd)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            print("[!] Failed to start via Docker.")
            print(stderr.read().decode())
            raise RuntimeError("Docker startup failed.")

        # Fetch PID (best-effort)
        get_pid_cmd = f"docker inspect spring-rest-service-spring-rest-service-1 | jq -r '.[0].State.Pid'"
        _, stdout, stderr = client.exec_command(get_pid_cmd)
        pid = stdout.read().decode().strip()
        err_output = stderr.read().decode().strip()

        if not pid.isdigit():
            print(f"[!] Could not retrieve PID for {compose_up_cmd}. stderr: {err_output}")

        pids: Dict[str, str] = {}

        print(f"    ↳ PID: {pid}")
        pids['spring-rest-service'] = pid

        # Persist for use in the runner (unchanged behavior)
        config['__springpids__'] = pids

        print("[✓] Docker started successfully.")
    finally:
        client.close()


def stop_docker(hostname: str, username: str, password: str, config: dict, output_dir: str) -> None:
    """
    Stop using Docker Compose.

    Reads the command from:
        config['experiment']['remote_docker_stop']
    """
    print("[~] Stopping Docker ...")
    compose_stop_cmd = config['experiment']['remote_docker_stop']
    compose_logs_cmd = config['experiment']['remote_docker_logs']

    client = _connect(hostname, username, password)
    try:
        # Erstelle Zeitstempel für den Dateinamen
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"docker_compose_logs_{timestamp}.txt"

        # Stelle sicher, dass das Verzeichnis existiert
        #output_dir = config['experiment']['local_output_directory']  # Hier Ihr gewünschtes Verzeichnis angeben
        os.makedirs(output_dir, exist_ok=True)

        # Kombiniere Verzeichnispfad mit Dateinamen
        output_path = os.path.join(output_dir, output_filename)

        _, stdout, stderr = client.exec_command(compose_logs_cmd)
        output_content = stdout.read().decode().strip()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)

        _, stdout, stderr = client.exec_command(compose_stop_cmd)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            print("[!] Failed to stop via Docker.")
            print(stderr.read().decode())
        else:
            print("[✓] Docker stopped.")
    finally:
        client.close()