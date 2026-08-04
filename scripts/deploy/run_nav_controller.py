#!/usr/bin/env python3
"""
Script for running the Navigation Controller.

This script receives observations from the robot and computes the velocity commands with the NavController.
It can be used when the robot is running with a **local policy** with the `--server` flag.

"""

import socket
import struct
import time
import torch
import numpy as np
import subprocess
import threading
import queue
import sys

import cv2
from go1_navigation.nav_controller import NavController

# Constants matching policy_runner.py
ACTION_MSG_LEN = 50  # 2 bytes (code) + 12*4 bytes (actions)
CMD_CODE_ACTION = 2

SERVER_IP = "192.168.123.14"  # "192.168.123.14", or use "localhost" if `policy_runner` is on the same machine
SERVER_PORT = 9292

LOOP_FREQ = 50  # 50 Hz
NAV_CONTROLLER_FREQ = 5  # 5 Hz


class SocketGStreamerCamera:
    """
    Launches GStreamer as an external process and reads raw bytes over a local TCP socket.
    """
    def __init__(self, cam_id=1, width=640, height=480, ip_last_segment="13"):
        self.width = width
        self.height = height
        self.cam_id = cam_id
        self.ip_last_segment = ip_last_segment

        self.tcp_port = 5000 + cam_id
        self.process = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None

    def _get_gstreamer_command(self):
        udp_ports = [9201, 9202, 9203, 9204, 9205]
        port = udp_ports[self.cam_id - 1]

        return [
            "gst-launch-1.0",
            "udpsrc", f"address=192.168.123.{self.ip_last_segment}", f"port={port}",
            "!", "application/x-rtp,media=video,encoding-name=H264",
            "!", "rtph264depay",
            "!", "h264parse",
            "!", "avdec_h264",
            "!", "videoconvert",
            "!", "videoscale",
            "!", f"video/x-raw,width={self.width},height={self.height},format=BGR",
            "!", "tcpserversink", "host=127.0.0.1", f"port={self.tcp_port}"
        ]

    def start(self):
        cmd = self._get_gstreamer_command()
        print(f"Launching external GStreamer process...")
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=0)
            self.running = True

            # Start background socket reader thread
            self.thread = threading.Thread(target=self._socket_reader_loop, daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            print(f"Failed to start GStreamer process: {e}")
            return False

    def _socket_reader_loop(self):
        frame_size = self.width * self.height * 3  # BGR = 3 bytes per pixel
        time.sleep(2.0)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", self.tcp_port))
            sock.settimeout(1.0)
            print(f"Successfully bridged to GStreamer via local TCP port {self.tcp_port}")

            while self.running:
                raw_frame = b""
                while len(raw_frame) < frame_size and self.running:
                    chunk = sock.recv(frame_size - len(raw_frame))
                    if not chunk:
                        print("GStreamer connection closed.")
                        return
                    raw_frame += chunk

                if len(raw_frame) != frame_size:
                    continue
                # Reconstruct raw bytes into image array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self.height, self.width, 3))

                # Update queue with newest frame
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)

            sock.close()
        except Exception as e:
            if self.running:
                print(f"Socket camera reader error: {e}")

    def read(self):
        """Non-blocking retrieval of the latest frame array."""
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def release(self):
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
        if self.thread:
            self.thread.join(timeout=1.0)


def load_policy(path):
    """Load the policy from a .pt file"""
    print(f"Loading policy from {path}...")
    policy = torch.jit.load(path, map_location="cpu")
    policy.eval()
    print("Policy loaded successfully!")
    return policy


def send_action_command(sock, actions):
    """Send action command to the robot.

    Args:
        sock: socket connection
        actions: list or array of 12 float values for joint actions
    """
    if len(actions) != 12:
        raise ValueError("Actions must contain exactly 12 values")

    # Pack: code (short) + 12 floats
    message = struct.pack("<h12f", CMD_CODE_ACTION, *actions)
    sock.send(message)


def send_command(sock, x, y, r):
    """Send velocity command to the robot."""
    code = 1  # Command code
    data = struct.pack("<hfff", code, x, y, r)
    sock.sendall(data)


def receive_observations(sock) -> torch.Tensor:
    """Receive observations from the robot."""
    try:
        sock.settimeout(0.1)  # 100ms timeout

        # Read observation count
        count_data = sock.recv(1)
        if not count_data:
            return None

        obs_count = struct.unpack("B", count_data)[0]

        # Read observations (each observation is 4 bytes float)
        obs_bytes = obs_count * 4
        obs_data = b""
        while len(obs_data) < obs_bytes:
            chunk = sock.recv(obs_bytes - len(obs_data))
            if not chunk:
                return None
            obs_data += chunk

        # Unpack observations
        observations = struct.unpack(f"{obs_count}f", obs_data)
        return torch.tensor(observations, dtype=torch.float32)
    except socket.timeout:
        return None
    except Exception as e:
        print(f"Error receiving observations: {e}")
        return None


def get_camera_data(camera_frame) -> dict:
    """Convert camera data to a dict for the NavController."""
    if camera_frame is not None:
        # Split side-by-side wide stereo stitched frame (1856x800) down the middle
        h, w, _ = camera_frame.shape
        left_bgr = camera_frame[:, :w // 2]

        # Convert to RGB layout required by tracking algorithms
        left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        distance_data = None

        return {"camera_rgb": left_rgb, "camera_distance": distance_data}
    else:
        return None


def get_observation_dict(obs: torch.Tensor) -> dict:
    """Convert observation tensor to a dictionary for easier access."""
    obs_dict = {
        "base_ang_vel": obs[:, 0:3].cpu().numpy().flatten(),
        "projected_gravity": obs[:, 3:6].cpu().numpy().flatten(),
        "velocity_commands": obs[:, 6:9].cpu().numpy().flatten(),
        "joint_pos": obs[:, 9:21].cpu().numpy().flatten(),
        "joint_vel": obs[:, 21:33].cpu().numpy().flatten(),
        "actions": obs[:, 33:45].cpu().numpy().flatten(),
    }
    return obs_dict


def main():
    host = SERVER_IP  # Change this to robot's IP when testing on real robot
    port = SERVER_PORT

    print(f"Connecting to robot at {host}:{port}...")

    cam = SocketGStreamerCamera(cam_id=1, width=640, height=480, ip_last_segment="163")
    if not cam.start():
        print("Critical Error: GStreamer pipeline failed to initiate.")
        sys.exit(1)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            print("Connected!")
            print("Starting nav controller")
            print("Press Ctrl+C to stop")

            # --- Frequencies
            loop_period = 1.0 / LOOP_FREQ
            nav_controller_count = LOOP_FREQ / NAV_CONTROLLER_FREQ

            # --- Initialize NavController
            camera_params = {
                "fx": 320.0,
                "fy": 320.0,
                "cx": 320.0,
                "cy": 240.0,
            }
            tag_size = 0.16

            nav_controller = NavController(
                camera_params=camera_params,
                tag_size=tag_size,  # Length of the black square in meters
                tag_family="tag36h11",
            )

            command = np.zeros(3, dtype=np.float32)  # Initialize command
            step_count = 0
            start_time = time.time()
            last_obs_time = time.time()

            while True:
                loop_start = time.time()

                # --- Observations
                # Receive observations from robot
                observations = receive_observations(sock)
                if observations is not None:
                    last_obs_time = time.time()
                    observations_dict = get_observation_dict(observations)

                    # update NavController with observations
                    nav_controller.update(observations_dict)
                else:
                    # No observations received - check if connection is still alive
                    if time.time() - last_obs_time > 1.0:
                        print("No observations received for 1 second - connection may be lost")
                        break

                # --- Camera Data Retrieval
                ret, frame = cam.read()
                if ret:
                    camera_dict = get_camera_data(frame)
                    if camera_dict is not None:
                        nav_controller.update(camera_dict)

                # --- Commands
                if step_count % nav_controller_count == 0:
                    command = nav_controller.get_command()
                    send_command(sock, *command)

                # Print status summary
                if step_count % 100 == 0:
                    elapsed = time.time() - start_time
                    freq = step_count / elapsed if elapsed > 0 else 0
                    print(f"Step {step_count}, Frequency: {freq:.1f} Hz, Command: [{command[0]:.3f}, {command[1]:.3f}, {command[2]:.3f}]")

                # Maintain 50 Hz loop rate
                step_count += 1

                loop_time = time.time() - loop_start
                sleep_time = loop_period - loop_time  # 50 Hz = 0.02 seconds
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except ConnectionRefusedError:
        print("Could not connect to robot. Make sure the robot is running with: python deploy.py --server")
    except KeyboardInterrupt:
        print(f"\nExternal policy control stopped after {step_count} steps")
    except Exception as e:
        print(f"Error during policy control: {e}")
    finally:
        print("Shutting down external processes and closing stream connections...")
        cam.release()


if __name__ == "__main__":
    main()

