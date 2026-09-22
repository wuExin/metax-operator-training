import os
import sys

import paramiko

host = os.environ["SSH_HOST"]
port = int(os.environ.get("SSH_PORT", "22"))
user = os.environ["SSH_USER"]
password = os.environ["SSH_PASS"]
command = sys.argv[1]
timeout = int(os.environ.get("SSH_TIMEOUT", "120"))

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(
    host,
    port=port,
    username=user,
    password=password,
    timeout=30,
    allow_agent=False,
    look_for_keys=False,
)

stdin, stdout, stderr = client.exec_command(command, timeout=timeout, get_pty=True)
out = stdout.read().decode("utf-8", "replace")
code = stdout.channel.recv_exit_status()
sys.stdout.write(out)
sys.stdout.write(f"\n[exit={code}]")
client.close()
sys.exit(code)
