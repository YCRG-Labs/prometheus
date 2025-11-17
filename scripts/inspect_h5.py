import h5py

def print_structure(name, obj):
    print(f"{name}: {type(obj).__name__}")
    if isinstance(obj, h5py.Dataset):
        print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")

files = ['data/ising_3d_small.h5', 'data/ising_processed_20250831_145828.h5', 'data/test_enhanced_3d_data.h5']

for filename in files:
    print(f"\n{'='*60}")
    print(f"File: {filename}")
    print('='*60)
    try:
        with h5py.File(filename, 'r') as f:
            f.visititems(print_structure)
    except Exception as e:
        print(f"Error: {e}")
