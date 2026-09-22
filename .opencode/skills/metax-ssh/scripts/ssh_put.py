import os

import paramiko

host = os.environ["SSH_HOST"]
port = int(os.environ.get("SSH_PORT", "22"))
user = os.environ["SSH_USER"]
password = os.environ["SSH_PASS"]

local, remote = __import__("sys").argv[1:3]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=port, username=user, password=password, timeout=30,
               allow_agent=False, look_for_keys=False)
sftp = client.open_sftp()
sftp.put(local, remote)
sftp.close()
client.close()
print(f"uploaded {local} -> {remote}")
