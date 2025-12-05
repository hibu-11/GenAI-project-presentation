"""
Data preprocessing for CodeSearchNet dataset
Prepares function-level code and documentation pairs for training

Note: Due to recent changes in Hugging Face datasets library,
this script includes a fallback to create synthetic demo data.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value
from tqdm import tqdm
import random

class CodeSearchNetPreprocessor:
    """Preprocesses CodeSearchNet data for documentation generation"""
    
    LANGUAGES = ['python', 'java', 'javascript', 'go', 'ruby', 'php']
    
    def __init__(self, output_dir: str = "data/processed", max_samples: int = None, use_synthetic: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.use_synthetic = use_synthetic
        
    def create_synthetic_data(self, language: str, num_samples: int = 100) -> List[Dict]:
        """Create synthetic data for demonstration purposes"""
        print(f"Creating {num_samples} synthetic samples for {language}...")
        
        synthetic_samples = {
            'python': [
                {
                    'code': 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
                    'documentation': 'Calculate the nth Fibonacci number using recursion. Returns the Fibonacci value for the given input n.',
                    'func_name': 'fibonacci'
                },
                {
                    'code': 'def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1',
                    'documentation': 'Perform binary search on a sorted array. Returns the index of target element or -1 if not found.',
                    'func_name': 'binary_search'
                },
                {
                    'code': 'def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)',
                    'documentation': 'Sort an array using the merge sort algorithm. Returns a new sorted array.',
                    'func_name': 'merge_sort'
                }
            ],
            'java': [
                {
                    'code': 'public static int factorial(int n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}',
                    'documentation': 'Calculate factorial of a number recursively. Returns n! for positive integers.',
                    'func_name': 'factorial'
                }
            ],
            'javascript': [
                {
                    'code': 'function isPalindrome(str) {\n    return str === str.split(\'\').reverse().join(\'\');\n}',
                    'documentation': 'Check if a string is a palindrome. Returns true if the string reads the same forwards and backwards.',
                    'func_name': 'isPalindrome'
                }
            ]
        }
        
        base_samples = synthetic_samples.get(language, synthetic_samples['python'])
        
        # Replicate samples to reach desired number
        data = []
        for i in range(num_samples):
            sample = base_samples[i % len(base_samples)].copy()
            sample['language'] = language
            # Add slight variation to avoid exact duplicates
            sample['func_name'] = f"{sample['func_name']}_{i}" if i >= len(base_samples) else sample['func_name']
            data.append(sample)
        
        return data
        
    def load_and_filter(self, language: str, split: str) -> List[Dict]:
        """Load and filter CodeSearchNet data with fallback to synthetic data"""
        print(f"Loading {language} {split} data...")
        
        # If synthetic data requested, skip real data loading
        if self.use_synthetic:
            if split == 'train':
                return self.create_synthetic_data(language, num_samples=100)
            elif split == 'validation':
                return self.create_synthetic_data(language, num_samples=20)
            else:  # test
                return self.create_synthetic_data(language, num_samples=20)
        
        # Try loading from Hugging Face (this may fail due to recent changes)
        try:
            from datasets import load_dataset
            dataset = load_dataset(
                "code_search_net",
                language,
                split=split
            )
        except Exception as e:
            print(f"Error loading {language} from Hugging Face: {e}")
            print(f"Falling back to synthetic data for demonstration...")
            if split == 'train':
                return self.create_synthetic_data(language, num_samples=100)
            elif split == 'validation':
                return self.create_synthetic_data(language, num_samples=20)
            else:
                return self.create_synthetic_data(language, num_samples=20)
        
        filtered_data = []
        for item in tqdm(dataset, desc=f"Filtering {language}"):
            # Filter out empty or very short code/docstrings
            if (item.get('func_code_string') and 
                item.get('func_documentation_string') and
                len(item['func_code_string'].strip()) > 50 and
                len(item['func_documentation_string'].strip()) > 10):
                
                filtered_data.append({
                    'code': item['func_code_string'].strip(),
                    'documentation': item['func_documentation_string'].strip(),
                    'language': language,
                    'func_name': item.get('func_name', 'unknown')
                })
                
                if self.max_samples and len(filtered_data) >= self.max_samples:
                    break
        
        return filtered_data
    
    def create_train_val_test_split(
        self, 
        data: List[Dict], 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train/val/test sets"""
        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        return (
            data[:train_size],
            data[train_size:train_size + val_size],
            data[train_size + val_size:]
        )
    
    def preprocess_all_languages(self):
        """Preprocess all languages and create combined dataset"""
        all_train, all_val, all_test = [], [], []
        
        for lang in self.LANGUAGES:
            print(f"\n{'='*60}")
            print(f"Processing {lang.upper()}")
            print('='*60)
            
            # Load train data (CodeSearchNet has pre-split data)
            train_data = self.load_and_filter(lang, 'train')
            val_data = self.load_and_filter(lang, 'validation')
            test_data = self.load_and_filter(lang, 'test')
            
            print(f"{lang}: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
            
            all_train.extend(train_data)
            all_val.extend(val_data)
            all_test.extend(test_data)
        
        # Check if we have any data
        if len(all_train) == 0:
            print("\n" + "="*60)
            print("WARNING: No data loaded from Hugging Face!")
            print("Creating synthetic demonstration dataset...")
            print("="*60)
            
            # Create synthetic data
            for lang in self.LANGUAGES[:3]:  # Just use first 3 languages for demo
                train_data = self.create_synthetic_data(lang, num_samples=100)
                val_data = self.create_synthetic_data(lang, num_samples=20)
                test_data = self.create_synthetic_data(lang, num_samples=20)
                
                all_train.extend(train_data)
                all_val.extend(val_data)
                all_test.extend(test_data)
                
                print(f"{lang}: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
        
        # Define the schema for the dataset
        features = Features({
            'code': Value('string'),
            'documentation': Value('string'),
            'language': Value('string'),
            'func_name': Value('string')
        })
        
        # Create Dataset objects with explicit features
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(pd.DataFrame(all_train), features=features),
            'validation': Dataset.from_pandas(pd.DataFrame(all_val), features=features),
            'test': Dataset.from_pandas(pd.DataFrame(all_test), features=features)
        })
        
        # Save to disk
        output_path = self.output_dir / "code_doc_dataset"
        dataset_dict.save_to_disk(str(output_path))
        
        # Save statistics
        stats = {
            'languages': self.LANGUAGES,
            'splits': {
                'train': len(all_train),
                'validation': len(all_val),
                'test': len(all_test)
            },
            'language_distribution': {},
            'data_source': 'synthetic_demo' if len(all_train) <= 1000 else 'codesearchnet'
        }
        
        for lang in self.LANGUAGES:
            lang_count = sum(1 for item in all_train if item['language'] == lang)
            stats['language_distribution'][lang] = lang_count
        
        with open(self.output_dir / "dataset_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n{'='*60}")
        print("PREPROCESSING COMPLETE")
        print('='*60)
        print(f"Total samples: {sum(stats['splits'].values())}")
        print(f"Data source: {stats['data_source']}")
        print(f"Saved to: {output_path}")
        print(f"Statistics saved to: {self.output_dir / 'dataset_stats.json'}")
        
        if stats['data_source'] == 'synthetic_demo':
            print("\n" + "="*60)
            print("NOTE: Using synthetic demonstration data")
            print("="*60)
            print("For production use, consider:")
            print("1. Downloading CodeSearchNet directly from GitHub")
            print("2. Using alternative datasets from Hugging Face")
            print("3. Creating your own dataset from public repositories")
            print("="*60)
        
        return dataset_dict, stats

def main():
    """Main preprocessing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess CodeSearchNet dataset')
    parser.add_argument('--synthetic', action='store_true', 
                       help='Use synthetic demo data instead of trying to download')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per language (for testing)')
    
    args = parser.parse_args()
    
    # For demo purposes, limit samples per language to 1000
    # Remove max_samples parameter for full dataset
    preprocessor = CodeSearchNetPreprocessor(
        output_dir="processed",
        max_samples=args.max_samples,
        use_synthetic=args.synthetic
    )
    
    dataset_dict, stats = preprocessor.preprocess_all_languages()
    
    # Display sample
    print("\n" + "="*60)
    print("SAMPLE DATA")
    print("="*60)
    sample = dataset_dict['train'][0]
    print(f"Language: {sample['language']}")
    print(f"Function: {sample['func_name']}")
    print(f"\nCode:\n{sample['code']}")
    print(f"\nDocumentation:\n{sample['documentation']}")
    print("\n" + "="*60)
    print("Setup complete! You can now:")
    print("1. Run the dashboard: streamlit run ../dashboard/app.py")
    print("2. Explore models in ../models/")
    print("3. Check the Jupyter notebook in ../notebooks/")
    print("="*60)

if __name__ == "__main__":
    main()