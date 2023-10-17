import torch
import time

print("Testing if CUDA supporting GPU is available...")
time.sleep(2)

if torch.cuda.is_available():
    print("GPU is available.")
    device = torch.device("cuda")
else:
    print("GPU is not available.")
    device = torch.device("cpu")
    
x = torch.randn(52000, 52000).to(device)

print("Loaded big array into VRAM; sleeping for 10 sec,",
      "please check nvidia-settings to see utilization!")
time.sleep(10)

print("Finished!")
