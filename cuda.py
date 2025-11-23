import torch
import gc

# Force garbage collection
gc.collect()

# Clear CUDA cache
torch.cuda.empty_cache()

# Reset peak memory stats (optional)
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_accumulated_memory_stats()

# Print memory status
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print("✓ CUDA cache cleared")