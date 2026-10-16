import paramiko
import sys

HOSTNAME = "127.0.0.1"
PORT = 2222
USERNAME = "root"
PASSWORD = "123456"

print(f"Connecting to Colab via local bridge {HOSTNAME}:{PORT}...")

try:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=HOSTNAME,
        port=PORT,
        username=USERNAME,
        password=PASSWORD,
        timeout=15
    )
    print("========================================")
    print("SUCCESS: Connected to Google Colab GPU!")
    print("========================================")
    
    # Run nvidia-smi
    stdin, stdout, stderr = client.exec_command("nvidia-smi")
    gpu_info = stdout.read().decode('utf-8')
    print("Remote GPU Info:\n" + gpu_info)
    
    # Run python environment check
    stdin, stdout, stderr = client.exec_command("python3 -c 'import torch; print(\"PyTorch:\", torch.__version__, \"| CUDA:\", torch.cuda.is_available(), \"| GPU:\", torch.cuda.get_device_name(0))'")
    env_info = stdout.read().decode('utf-8')
    print("PyTorch Environment:\n" + env_info)
    
    client.close()
except Exception as e:
    print(f"Connection failed: {e}", file=sys.stderr)
