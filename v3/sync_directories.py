import os
import time
import paramiko
import configparser
from scp import SCPClient

# Configuration
filepath = os.path.abspath(__file__)
parent_dir = os.path.dirname(filepath)
project_dir = os.path.dirname(parent_dir)
config_file = os.path.join(project_dir, "config.ini")
print(f"File Path: {filepath}")
print(f"Parent Dir: {parent_dir}")
print(f"Project Dir: {project_dir}")
print(f"Config File: {config_file}")
config = configparser.ConfigParser()
config.sections()
config.read(config_file)
REMOTE_HOST = config['GLOBAL']['RemoteHost']
REMOTE_USERNAME = config['GLOBAL']['RemoteUserName']
PASSWORD = config['GLOBAL']['password']
REMOTE_DIRS = [
    config['DIRECTORY_SYNC']['r_dir1'], 
    config['DIRECTORY_SYNC']['r_dir2'],
    config['DIRECTORY_SYNC']['r_dir3'],
    config['DIRECTORY_SYNC']['r_dir4']
]
LOCAL_DIRS = [
    config['DIRECTORY_SYNC']['dir1'], 
    config['DIRECTORY_SYNC']['dir2'],
    config['DIRECTORY_SYNC']['dir3'],
    config['DIRECTORY_SYNC']['dir4']
]

def get_ssh_client(host, username, password):
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=host, username=username, password=password)
    except ValueError:
        raise ValueError("PASSWORD MISSING: GO TYPE IN PASSWORD ON CONFIG.INI FILE")
    return client

def list_remote_files(ssh, remote_dir):
    stdin, stdout, stderr = ssh.exec_command(f'ls -1 "{remote_dir}"')
    files = stdout.read().decode().splitlines()
    return set(files)

def list_local_files(local_dir):
    return set(os.listdir(local_dir))

def sync_missing_files(ssh, scp, local_dir, remote_dir, remote_files):
    local_files = list_local_files(local_dir)
    missing_files = local_files - remote_files
    print(f"Len Missing Files: {len(missing_files)}")

    x = 1
    for file in missing_files:
        try:
            local_path = os.path.join(local_dir, file)
            print(f"Uploading {x} of {len(missing_files)} missing files: {file} ")
            scp.put(local_path, os.path.join(remote_dir, file))
        except Exception:
            print("Something went wrong, in sync_missing_files...")
        else:
            x += 1

def main():
    ssh = get_ssh_client(REMOTE_HOST, REMOTE_USERNAME, PASSWORD)
    scp = SCPClient(ssh.get_transport())

    try:
        for remote_dir, local_dir in zip(REMOTE_DIRS, LOCAL_DIRS):
            print(f"\nChecking: {local_dir} ➝ {remote_dir}")
            remote_files = list_remote_files(ssh, remote_dir)
            sync_missing_files(ssh, scp, local_dir, remote_dir, remote_files)

    finally:
        scp.close()
        ssh.close()

if __name__ == "__main__":
    main()
