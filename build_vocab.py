"""Build a vocabulary file."""

import argparse
import os
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    filenames = os.listdir(args.data_dir)
    file_paths = [os.path.join(args.data_dir, filename)
                  for filename in sorted(filenames)]
    counter = Counter()
    for file_path in file_paths:
        print(f'Processing {file_path}...')
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                counter.update(line.split())

    print(f'Complete!')
    print(f'# unique words: {len(counter)}')
    with open(args.out, 'w', encoding='utf-8') as f:
        for w, c in counter.most_common():
            f.write(f'{w}\t{c}\n')
    print(f'Saved into {args.out}')


if __name__ == '__main__':
    main()
