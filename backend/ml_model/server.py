"""
EfficientNet Pneumonia Detection Server

This script serves the pneumonia detection model from the ../pneumonia-ml-efficientnet folder
as a Flask API for the Next.js frontend to use.
"""

import os
import sys
import shutil
import subprocess
import time
import signal
import argparse

# Add parent directory to sys.path to allow importing from parent directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def start_efficientnet_server(model_path=None, port=5000, host='0.0.0.0'):
    """
    Start the EfficientNet model server using the app.py from pneumonia-ml-efficientnet
    
    Args:
        model_path (str, optional): Path to specific model file. If None, uses default.
        port (int, optional): Port to run the server on. Defaults to 5000.
        host (str, optional): Host to bind to. Defaults to '0.0.0.0'.
    """
    print("Starting EfficientNet Pneumonia Detection Server...")
    
    # Get the path to the pneumonia-ml-efficientnet directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(project_root, 'pneumonia-ml-efficientnet')
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found at {model_dir}")
        return False
        
    # Check if app.py exists
    app_path = os.path.join(model_dir, 'app.py')
    if not os.path.exists(app_path):
        print(f"Error: app.py not found at {app_path}")
        return False
    
    # Construct environment variables
    env = os.environ.copy()
    
    # If model_path is not provided, explicitly use the best_model.pth from output directory
    if not model_path:
        output_dir = os.path.join(project_root, 'output')
        model_path = os.path.join(output_dir, 'best_model.pth')
        print(f"Using model from output directory: {model_path}")
        
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
        
    # Set the MODEL_PATH environment variable
    env['MODEL_PATH'] = model_path
    print(f"Setting MODEL_PATH to: {model_path}")
    
    # Start the Flask app in a subprocess
    try:
        # Build the command to run
        command = [
            sys.executable, 
            app_path,
            '--host', host,
            '--port', str(port)
        ]
        
        # Start the process
        print(f"Running command: {' '.join(command)}")
        process = subprocess.Popen(
            command,
            env=env,
            cwd=model_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Wait a bit for startup
        time.sleep(2)
        
        if process.poll() is not None:
            # Process has already terminated
            stdout, stderr = process.communicate()
            print("Server failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
        
        print(f"EfficientNet server started on http://{host}:{port}")
        return process
        
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        return False

def stop_server(process):
    """Stop the Flask server process"""
    if process:
        print("Stopping server...")
        process.send_signal(signal.SIGTERM)
        process.wait()
        print("Server stopped")

def main():
    parser = argparse.ArgumentParser(description="Start the EfficientNet Pneumonia Detection Server")
    parser.add_argument('--model-path', type=str, help='Path to the model file')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    # Start server and keep it running
    server_process = start_efficientnet_server(args.model_path, args.port, args.host)
    
    if not server_process:
        print("Failed to start server")
        sys.exit(1)
    
    try:
        # Keep the script running and monitor the server
        print("Server is running. Press Ctrl+C to stop.")
        while True:
            # Check if server is still running
            if server_process.poll() is not None:
                stdout, stderr = server_process.communicate()
                print("Server stopped unexpectedly:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                
                # Try to restart the server
                print("Restarting server...")
                server_process = start_efficientnet_server(args.model_path, args.port, args.host)
                if not server_process:
                    print("Failed to restart server")
                    sys.exit(1)
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_server(server_process)

if __name__ == "__main__":
    main() 