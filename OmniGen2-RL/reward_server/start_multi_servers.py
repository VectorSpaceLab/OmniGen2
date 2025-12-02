#!/usr/bin/env python3
"""
Script to launch multi-GPU servers.
Start specified server instances on all GPUs.
"""

import subprocess
import time
import os
import signal
import sys
import argparse
from multiprocessing import Process
import yaml

# Store subprocesses
server_processes = []

def start_server(args, worker_idx, num_gpus_per_worker, port, server_script, unknown_args, log_name):
    """Start server on the specified GPU(s)"""
    cmd = [
        sys.executable, server_script,
        "--port", str(port),
        "--config_path", args.config_path,
        *unknown_args,
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(worker_idx * num_gpus_per_worker, (worker_idx + 1) * num_gpus_per_worker))
    
    # Create log directory
    os.makedirs("./logs", exist_ok=True)
    log_file = f"./logs/{log_name}_worker_{worker_idx}.log"
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
        print(f"📝 Worker {worker_idx} log file: {log_file}")
        return process
    except Exception as e:
        print(f"❌ Failed to start server for Worker {worker_idx}: {e}")
        return None

def signal_handler(signum, frame):
    """Handle exit signals"""
    print("\n🛑 Received exit signal, shutting down all servers...")
    for i, process in enumerate(server_processes):
        if process and process.poll() is None:
            print(f"Shutting down server Worker {i}")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    sys.exit(0)

def check_server_logs(args, worker_idx):
    """Check server log file"""
    log_file = f"./logs/{args.log_name}_worker_{worker_idx}.log"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                if content:
                    print(f"\n📄 Worker {worker_idx} error log:")
                    print(content[-1000:])  # Show last 1000 characters
        except Exception as e:
            print(f"Failed to read log file {log_file}: {e}")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Launch multi-GPU servers')
    parser.add_argument('server_script', help='Server script filename to launch')
    parser.add_argument('--config_path', type=str, required=True, help='Config path')
    parser.add_argument('--machine_id', type=int, default=0, help='Machine id')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    args, unknown_args = parser.parse_known_args()

    log_name = f"reward_server_{args.model_name}_machine_{args.machine_id}"

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    hosts = config["server"]["hosts"]
    worker_base_port = config["server"]["worker_base_port"]
    num_gpus_per_worker = config["reward"]["tensor_parallel_size"]
    
    # Add .py suffix if missing
    server_script = args.server_script
    if not server_script.endswith('.py'):
        server_script += '.py'
    
    # Check if script exists
    if not os.path.exists(server_script):
        print(f"❌ Script file {server_script} does not exist")
        sys.exit(1)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Detect available GPUs
    import torch
    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
    else:
        num_available_gpus = 0
    
    num_workers = num_available_gpus // num_gpus_per_worker
    
    if num_workers == 0:
        print("❌ No available GPUs, exiting")
        sys.exit(1)
        
    print(f"🔥 Launching {num_workers} {server_script} server(s)")
    print(f"📍 Base port: {worker_base_port}")
    print(f"🖥️  Host: {hosts[args.machine_id]}")
    print("-" * 50)
    
    # Start all servers
    for worker_idx in range(num_workers):
        port = worker_base_port + worker_idx
        process = start_server(args, worker_idx, num_gpus_per_worker, port, server_script, unknown_args, log_name)
        server_processes.append(process)
        time.sleep(3)  # Add delay to avoid resource contention
    
    print(f"\n✅ Attempted to launch {num_workers} server(s)")
    print("Server list:")
    for worker_idx in range(num_workers):
        port = worker_base_port + worker_idx
        print(f"  Server {worker_idx}: http://{hosts[args.machine_id]}:{port}")
    
    print("\n⏳ Waiting for servers to start...")
    time.sleep(10)  # Wait for servers to start
    
    # Check server status
    running_servers = 0
    for i, process in enumerate(server_processes):
        if process and process.poll() is None:
            running_servers += 1
            print(f"✅ Server {i} is running")
        else:
            print(f"❌ Server {i} failed to start")
            check_server_logs(args, i)
    
    if running_servers == 0:
        print("❌ No servers started successfully")
        sys.exit(1)
    
    print(f"\n🎉 Successfully started {running_servers} server(s)")
    print("⏳ Servers are running... (Press Ctrl+C to exit)")
    
    # Monitor server status
    try:
        while True:
            time.sleep(10)  # Check interval
            # Check for crashed servers
            for i, process in enumerate(server_processes):
                if process and process.poll() is not None:
                    print(f"⚠️  Server {i} exited (return code: {process.returncode})")
                    check_server_logs(args, i)
                    # Set exited process to None to avoid duplicate reporting
                    server_processes[i] = None
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main() 