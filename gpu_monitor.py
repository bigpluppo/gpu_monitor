import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from datetime import datetime
from PIL import Image
import io
import requests
from pathlib import Path
import os

class GPUMonitor:
    def __init__(self, update_interval=0.25):
        # Set dark theme for matplotlib
        plt.style.use('dark_background')
        
        # Create figure with grid specification for custom layout
        self.fig = plt.figure(figsize=(15, 10), facecolor='black')
        gs = self.fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.3)
        
        # Create subplots
        self.axs = [
            self.fig.add_subplot(gs[0, 0]),  # Utilization
            self.fig.add_subplot(gs[1, 0]),  # Temperature
            self.fig.add_subplot(gs[2, 0]),  # Memory
            self.fig.add_subplot(gs[0, 1]),  # Utilization GIF
            self.fig.add_subplot(gs[1, 1]),  # Temperature GIF
            self.fig.add_subplot(gs[2, 1]),  # Memory GIF
        ]
        
        self.fig.suptitle('GPU MONITOR-Z100', color='#00ff00', fontsize=16, fontweight='bold')
        
        # Initialize data storage
        self.timestamps = []
        self.utilization = []
        self.temperature = []
        self.memory_used = []
        self.memory_total = []
        self.update_interval = update_interval
        self.max_points = 60
        
        # Load GIFs
        self.load_gifs()
        
        # Style configuration
        self.line_color = '#00ff00'  # Hacker green
        self.grid_color = '#1a1a1a'  # Dark gray
        self.text_color = '#00ff00'
        
    def download_gif(self, url, filename):
        """Download a GIF if it doesn't exist locally"""
        if not os.path.exists(filename):
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return True
        return os.path.exists(filename)

    def extract_gif_frames(self, gif_path):
        """Extract frames from a GIF file"""
        frames = []
        try:
            with Image.open(gif_path) as gif:
                for frame_idx in range(gif.n_frames):
                    gif.seek(frame_idx)
                    # Convert to RGB to ensure consistent format
                    frame = gif.convert('RGB')
                    # Convert to numpy array and normalize
                    frame_array = np.array(frame) / 255.0
                    frames.append(frame_array)
        except Exception as e:
            print(f"Error loading GIF {gif_path}: {e}")
            # Create a fallback frame if GIF loading fails
            frames = [np.zeros((100, 100, 3)) for _ in range(10)]
            for i, frame in enumerate(frames):
                frame[:, :, 1] = i / 10  # Green channel gradient
        return frames

    def load_gifs(self):
        """Load all GIF files for the visualizations"""
        # Create cache directory if it doesn't exist
        cache_dir = Path("gif_cache")
        cache_dir.mkdir(exist_ok=True)

        # GIF URLs - using cyberpunk/tech themed GIFs
        gif_urls = {
            'utilization': 'goku.gif',
            'temperature': 'goku.gif',
            'memory': 'goku.gif'
        }

        try:
            # First try to load from cache directory
            util_path = cache_dir / "utilization.gif"
            temp_path = cache_dir / "temperature.gif"
            mem_path = cache_dir / "memory.gif"

            # If files don't exist, use fallback logic to create frames
            if not all(path.exists() for path in [util_path, temp_path, mem_path]):
                print("GIF files not found, creating fallback animations...")
                self.create_fallback_gifs(cache_dir)

            # Load the frames
            self.util_frames = self.extract_gif_frames(util_path)
            self.temp_frames = self.extract_gif_frames(temp_path)
            self.mem_frames = self.extract_gif_frames(mem_path)

        except Exception as e:
            print(f"Error during GIF loading: {e}")
            self.create_fallback_gifs(cache_dir)

    def create_fallback_gifs(self, cache_dir):
        """Create fallback GIF files with simple animations"""
        for name in ['utilization', 'temperature', 'memory']:
            frames = []
            for i in range(10):
                # Create a gradient image with cyberpunk colors
                img = Image.new('RGB', (100, 100))
                pixels = img.load()
                for x in range(100):
                    for y in range(100):
                        # Create a gradient with some variation
                        green = int((i / 10.0) * 255)
                        blue = int((x / 100.0) * 128)
                        pixels[x, y] = (0, green, blue)
                frames.append(img)
            
            # Save as GIF
            frames[0].save(
                cache_dir / f"{name}.gif",
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0
            )
            
    def get_gpu_info(self):
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total', 
                 '--format=csv,noheader,nounits'],
                encoding='utf-8'
            )
            util, temp, mem_used, mem_total = map(float, output.strip().split(','))
            return util, temp, mem_used, mem_total
        except Exception as e:
            print(f"Error reading GPU metrics: {e}")
            return 0, 0, 0, 0
            
    def get_frame_index(self, value, max_value, frames, min_value=0):
        """Convert a value to a frame index"""
        normalized = max(min((value-min_value) / (max_value-min_value), 1.0),0)
        return int(normalized * (len(frames) - 1))
            
    def update(self, frame):
        # Get current GPU metrics
        util, temp, mem_used, mem_total = self.get_gpu_info()
        current_time = datetime.now()
        
        # Update data lists
        self.timestamps.append(current_time)
        self.utilization.append(util)
        self.temperature.append(temp)
        self.memory_used.append(mem_used)
        self.memory_total.append(mem_total)
        
        # Keep only recent history
        if len(self.timestamps) > self.max_points:
            self.timestamps.pop(0)
            self.utilization.pop(0)
            self.temperature.pop(0)
            self.memory_used.pop(0)
            self.memory_total.pop(0)
        
        # Clear all subplots
        for ax in self.axs:
            ax.clear()
        
        # Style settings for all plots
        plot_kwargs = {
            'color': self.line_color,
            'linewidth': 2,
        }
        
        # Plot GPU Utilization
        self.axs[0].plot(self.timestamps, self.utilization, **plot_kwargs)
        self.axs[0].set_title('GPU UTILIZATION (%)', color=self.text_color, pad=10)
        self.axs[0].set_ylim(0, 100)
        self.axs[0].grid(True, color=self.grid_color)
        self.axs[0].tick_params(colors=self.text_color)
        
        # Plot Temperature
        self.axs[1].plot(self.timestamps, self.temperature, **plot_kwargs)
        self.axs[1].set_title('TEMPERATURE (C)', color=self.text_color, pad=10)
        self.axs[1].set_ylim(0, 100)
        self.axs[1].grid(True, color=self.grid_color)
        self.axs[1].tick_params(colors=self.text_color)
        
        # Plot Memory Usage
        self.axs[2].plot(self.timestamps, self.memory_used, **plot_kwargs)
        self.axs[2].set_title('MEMORY USAGE (MB)', color=self.text_color, pad=10)
        self.axs[2].set_ylim(0, mem_total)
        self.axs[2].grid(True, color=self.grid_color)
        self.axs[2].tick_params(colors=self.text_color)
        
        # Update GIF frames
        util_frame = self.util_frames[self.get_frame_index(util, 100, self.util_frames)]
        temp_frame = self.temp_frames[self.get_frame_index(temp, 100, self.temp_frames,min_value=25)]
        mem_frame = self.mem_frames[self.get_frame_index(mem_used, mem_total, self.mem_frames)]
        
        # Display frames
        self.axs[3].imshow(util_frame)
        self.axs[4].imshow(temp_frame)
        self.axs[5].imshow(mem_frame)
        
        # Remove axes from GIF displays
        for ax in self.axs[3:]:
            ax.axis('off')
        
        # Add current values as text on GIF frames
        self.axs[3].text(0.5, -0.1, f'{util:.1f}%', 
                        ha='center', va='center', color=self.text_color, 
                        transform=self.axs[3].transAxes)
        self.axs[4].text(0.5, -0.1, f'{temp:.1f}°C', 
                        ha='center', va='center', color=self.text_color, 
                        transform=self.axs[4].transAxes)
        self.axs[5].text(0.5, -0.1, f'{mem_used:.0f}/{mem_total:.0f} MB', 
                        ha='center', va='center', color=self.text_color, 
                        transform=self.axs[5].transAxes)
        
        # Adjust layout
        plt.tight_layout()
            
    def start(self):
        ani = FuncAnimation(self.fig, self.update, interval=self.update_interval * 1000)
        plt.show()

if __name__ == "__main__":
    monitor = GPUMonitor(update_interval=1)
    monitor.start()