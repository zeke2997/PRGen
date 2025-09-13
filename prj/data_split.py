import json
from collections import defaultdict
from pathlib import Path
import random

def split_dataset(input_file: str, output_dir: str, 
                  train_ratio: float = 0.8, 
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  seed: int = 42):

    random.seed(seed)
    
    print("=" * 80)
    print("PRGen Dataset Splitting (Within-Group Split)")
    print("=" * 80)
    
    print(f"\nLoading data from {input_file}...")
    grouped_data = defaultdict(list)
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            req_info = data['req_info']
            group_key = (req_info['protocol'], req_info['port'], req_info['req_name'])
            grouped_data[group_key].append(line.strip())
    
    print(f"Total groups: {len(grouped_data)}")
    total_samples = sum(len(v) for v in grouped_data.values())
    print(f"Total records: {total_samples:,}")
    
    print(f"\nGroup statistics:")
    for group_key, samples in sorted(grouped_data.items(), key=lambda x: len(x[1]), reverse=True):
        protocol, port, path = group_key
        print(f"  ({protocol}, {port}, {path}): {len(samples):,} samples")
    
    print(f"\n{'─' * 80}")
    print("Splitting each group into 80/10/10...")
    print(f"{'─' * 80}")
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    for group_key, samples in grouped_data.items():
        samples_copy = samples.copy()
        random.shuffle(samples_copy)
        
        n_samples = len(samples_copy)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        group_train = samples_copy[:n_train]
        group_val = samples_copy[n_train:n_train + n_val]
        group_test = samples_copy[n_train + n_val:]
        
        train_samples.extend(group_train)
        val_samples.extend(group_val)
        test_samples.extend(group_test)
        
        protocol, port, path = group_key
        print(f"  ({protocol}, {port}, {path[:30]}...):")
        print(f"    Train: {len(group_train):6,}  Val: {len(group_val):6,}  Test: {len(group_test):6,}")
    
    print(f"\n{'─' * 80}")
    print("Overall split:")
    print(f"  Train: {len(train_samples):8,} samples ({len(train_samples)/total_samples*100:.1f}%)")
    print(f"  Val:   {len(val_samples):8,} samples ({len(val_samples)/total_samples*100:.1f}%)")
    print(f"  Test:  {len(test_samples):8,} samples ({len(test_samples)/total_samples*100:.1f}%)")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    print(f"\n{'─' * 80}")
    print("Writing files...")
    print(f"{'─' * 80}")
    
    for split_name, samples in splits.items():
        output_file = output_dir / f"{split_name}_dataset.json"
        
        samples_copy = samples.copy()
        random.shuffle(samples_copy)
        
        with open(output_file, 'w') as f:
            for line in samples_copy:
                f.write(line + '\n')
        
        print(f"{split_name.capitalize()} set:")
        print(f"  File:    {output_file}")
        print(f"  Records: {len(samples):,}")
    
    print("\n" + "=" * 80)
    print(" Dataset splitting completed!")
    print("=" * 80)
    print("\nNote: Each group (protocol, port, path) has been split 80/10/10 internally.")
    print("This ensures all groups are represented in train/val/test sets.")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Split PRGen dataset within each group')
    parser.add_argument('--input', type=str, default='dataset/all_dataset.json')
    parser.add_argument('--output-dir', type=str, default='dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    split_dataset(args.input, args.output_dir, 
                  train_ratio=args.train_ratio,
                  val_ratio=args.val_ratio, 
                  test_ratio=args.test_ratio,
                  seed=args.seed)
