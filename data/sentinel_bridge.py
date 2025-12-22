import pandas as pd
import numpy as np
import torch

class SentinelBridge:
    """
    The Interface between Linux Kernel (Syscalls) and Weightless Neural Net (Binary).
    
    Pipeline:
    1. Stream of Syscall Numbers (e.g., [59, 12, 1, ...])
    2. Sliding Window (e.g., 100 syscalls)
    3. Histogram (Counts of each syscall type)
    4. Thermometer Encoding (Counts -> Binary Strings)
    """
    
    def __init__(self, window_size=100, max_syscall=335, thermometer_resolution=8):
        """
        Args:
            window_size: Number of syscalls per analysis window.
            max_syscall: Max syscall ID on x86_64 (approx 335).
            thermometer_resolution: Bits used to represent the frequency of EACH syscall.
                                    If resolution=8, max detectable count is 8.
        """
        self.window_size = window_size
        self.max_syscall = max_syscall
        self.resolution = thermometer_resolution
        
        # Total input bits = (Num Syscalls) * (Bits per count)
        # e.g., 336 * 8 = 2688 input bits per sample
        self.total_input_bits = (self.max_syscall + 1) * self.resolution

    def encode_thermometer(self, histogram):
        """
        Converts integer counts to thermometer code.
        Count 2, Res 4 -> [1, 1, 0, 0]
        Count 0, Res 4 -> [0, 0, 0, 0]
        """
        # Clip counts to max resolution
        counts = np.clip(histogram, 0, self.resolution)
        
        # Vectorized Thermometer Encoding
        # Shape: [Num_Syscalls, Resolution]
        # We create a matrix where index < count is 1, else 0
        range_matrix = np.arange(self.resolution).reshape(1, -1) # [0, 1, 2, ... R-1]
        counts_matrix = counts.reshape(-1, 1)                    # Column vector
        
        # Broadcasting comparison
        # If count is 3: [0, 1, 2] < 3 -> [T, T, T, F, F...]
        binary_matrix = (range_matrix < counts_matrix).astype(np.float32)
        
        return binary_matrix.flatten()

    def process_log(self, file_path):
        """
        Ingests a sentinel_log.csv and returns a Binary Tensor for DWN.
        """
        print(f"🔄 BRIDGE: Processing {file_path}...")
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"❌ ERROR: Could not read log. {e}")
            return None, None

        if df.empty:
            print("⚠️ WARNING: Log is empty.")
            return None, None

        # Group by PID to separate process traces
        grouped = df.groupby('pid')['syscall_nr'].apply(list)
        
        binary_samples = []
        labels = [] # To be filled if we have labelled data (Normal=0)

        for pid, trace in grouped.items():
            if len(trace) < self.window_size:
                continue
                
            # Sliding Window
            # Step size = window_size // 2 (50% overlap)
            step = self.window_size // 2
            
            for i in range(0, len(trace) - self.window_size, step):
                window = trace[i : i + self.window_size]
                
                # Compute Histogram (Bag of Syscalls)
                # minlength ensures we cover all 0..335 syscalls even if not present
                hist = np.bincount(window, minlength=self.max_syscall + 1)
                
                # Trim if bincount expanded beyond max_syscall (rare custom syscalls)
                if len(hist) > self.max_syscall + 1:
                    hist = hist[:self.max_syscall + 1]
                
                # Binarize
                binary_vec = self.encode_thermometer(hist)
                binary_samples.append(binary_vec)
                
                # For now, we assume this is "Normal" training data (Label 0)
                labels.append(0) 

        if not binary_samples:
            print("⚠️ WARNING: No valid windows found (traces too short?).")
            return None, None

        x_tensor = torch.tensor(np.array(binary_samples), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(labels), dtype=torch.long)
        
        print(f"✅ BRIDGE: Generated {x_tensor.shape[0]} samples. Input Dim: {x_tensor.shape[1]} bits.")
        return x_tensor, y_tensor