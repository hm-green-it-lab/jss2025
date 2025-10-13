# helper/jmeter.py
"""
Remote JMeter runner, fetcher, and shutdown helpers.

Highlights
----------
- Filenames follow the common pattern:
    {tool}_{experiment_type}_{YYYYMMDD_%H%M%S}_{iter}_{total}.{ext}
  e.g., configeter_teastore_tomcat_idle_20250819_110131_1_3.jtl

- Public function signatures and behavior are unchanged (call sites unaffected).
"""

from __future__ import annotations

import os
from stat import S_ISDIR
from datetime import datetime

import paramiko
import zipfile

def fetch_joularjx_artifacts(
    config: dict,
    local_output_root: str,
    cleanup_remote: bool = True
) -> None:
    sut_host = config.get("target_host")
    remote_dir  = config.get("remote_dir")
    joularjx_result_dir  = config.get("joularjx_result_dir")
    joularjx_zip_dir  = config.get("joularjx_zip_dir")

    if not sut_host or not remote_dir:
        print("[JMeter][fetch] Missing target_host/remote_dir; skip fetch.")
        return

    j_user = os.environ.get("SUT_SSH_USER")
    j_pass = os.environ.get("SUT_SSH_PASSWORD")
    if not j_user or not j_pass:
        print("[JoularJX][fetch] Missing SSH creds; skip fetch.")
        return

    # Determine destination folder based on the dt used for filenames
    dt = config.get("__jmeter_dt__") or _ts("%Y%m%d_%H%M%S")
    day_tag, time_tag = dt.split("_", 1)

    # Use the orchestrator-provided root as-is (absolute or relative)
    dest_dir = os.path.join(local_output_root, f"joularjx-result_{time_tag}")

    print(f"[JoularJX][fetch] Downloading artifacts to: {dest_dir}")

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=sut_host, username=j_user, password=j_pass)

        ssh.exec_command(f"sudo chown -R {j_user} {joularjx_result_dir}")
        print(f"[JoularJX][fetch] Changed Owner: {joularjx_result_dir}")

        zip_file = f"{joularjx_zip_dir}/joularjx-result_{time_tag}.zip"
        ssh.exec_command(f"cd {joularjx_result_dir} && zip -r {zip_file} .")
        print(f"[JoularJX][fetch] Created ZIP for download: {zip_file}")

        download_and_cleanup_sftp_recursive(
             hostname=sut_host,
             username=j_user,
             password=j_pass,
             joularjx_result_dir=joularjx_zip_dir,
             local_output_dir=dest_dir,
             cleanup_remote=True
         )


        # Lösche Remote-Datei falls cleanup_remote aktiviert ist
        if cleanup_remote:
            ssh.exec_command(f"rm -rf {joularjx_result_dir}/*")
            print(f"[JoularJX][fetch] Deleted content from joularjx: {joularjx_result_dir}")

        # Extract the ZIP file
        with zipfile.ZipFile(dest_dir + f"\\joularjx-result_{time_tag}.zip", 'r') as zip_ref:
            # Extract all files to a directory
            zip_ref.extractall(dest_dir)

        # Delete the ZIP file after extraction
        os.remove(dest_dir + f"\\joularjx-result_{time_tag}.zip")
    finally:
        ssh.close()

def download_and_cleanup_sftp_recursive(hostname, username, password, joularjx_result_dir, local_output_dir, cleanup_remote=False):
    try:
        # SFTP-Verbindung aufbauen
        transport = paramiko.Transport((hostname, 22))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # Stelle sicher, dass das lokale Verzeichnis existiert
        os.makedirs(local_output_dir, exist_ok=True)

        def process_directory(remote_dir, local_dir):
            """Rekursive Funktion zum Verarbeiten von Verzeichnissen"""
            try:
                # Liste alle Einträge im aktuellen Remote-Verzeichnis
                items = sftp.listdir_attr(remote_dir)

                for item in items:
                    remote_path = os.path.join(remote_dir, item.filename).replace("\\", "/")
                    local_path = os.path.join(local_dir, item.filename)

                    if S_ISDIR(item.st_mode):
                        # Wenn es ein Verzeichnis ist, erstelle es lokal und verarbeite rekursiv
                        print(f"Verarbeite Verzeichnis: {remote_path}")
                        os.makedirs(local_path, exist_ok=True)
                        process_directory(remote_path, local_path)

                        # Lösche leeres Remote-Verzeichnis nach Verarbeitung wenn cleanup aktiviert
                        if cleanup_remote:
                            sftp.rmdir(remote_path)
                    else:
                        # Wenn es eine Datei ist, lade sie herunter
                        print(f"Lade herunter: {remote_path}")
                        sftp.get(remote_path, local_path)

                        # Lösche Remote-Datei falls cleanup_remote aktiviert ist
                        if cleanup_remote:
                            print(f"Lösche Remote-Datei: {remote_path}")
                            sftp.remove(remote_path)

            except Exception as e:
                print(f"Fehler beim Verarbeiten von Verzeichnis {remote_dir}: {str(e)}")
                raise

        try:
            # Starte den rekursiven Download-Prozess
            process_directory(joularjx_result_dir, local_output_dir)

            if cleanup_remote:
                print(f"Alle Dateien aus {joularjx_result_dir} wurden rekursiv heruntergeladen und gelöscht")
            else:
                print(f"Alle Dateien aus {joularjx_result_dir} wurden rekursiv heruntergeladen")

        except Exception as e:
            print(f"Fehler beim Verarbeiten der Dateien: {str(e)}")
            raise

    except Exception as e:
        print(f"Fehler beim Verbindungsaufbau: {str(e)}")
        raise

    finally:
        # Verbindung schließen
        if 'sftp' in locals():
            sftp.close()
        if 'transport' in locals():
            transport.close()


def _ts(fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Return current local time formatted as string."""
    return datetime.now().strftime(fmt)