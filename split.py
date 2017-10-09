"""
Split data file(s) into multiple shards.
This is because a corpus is usually too large to be loaded into
the main memory. Using the multiple shards, we will mimic data
shuffling.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--num-shards', type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.save_dir)

    # Count the number of data lines
    num_lines = 0
    for filename in sorted(os.listdir(args.data_dir)):
        file_path = os.path.join(args.data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            file_num_lines = 0
            for _ in f:
                file_num_lines += 1
            print(f'{file_path}: {file_num_lines} lines')
            num_lines += file_num_lines

    # Save!
    lines_per_shard = num_lines // args.num_shards
    processed = 0
    cur_shard_id = -1
    cur_shard_file = None
    num_lines_in_cur_shard = 0
    for filename in sorted(os.listdir(args.data_dir)):
        file_path = os.path.join(args.data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                shard_id = processed // lines_per_shard
                if shard_id >= args.num_shards:
                    shard_id = args.num_shards - 1
                if shard_id != cur_shard_id:
                    print(f'#{cur_shard_id}: {num_lines_in_cur_shard} lines')
                    num_lines_in_cur_shard = 0
                    cur_shard_id = processed // lines_per_shard
                    cur_shard_filename = f'{cur_shard_id:05d}.txt'
                    cur_shard_file_path = os.path.join(
                        args.save_dir, cur_shard_filename)
                    if cur_shard_file:
                        cur_shard_file.close()
                    cur_shard_file = open(
                        cur_shard_file_path, 'w', encoding='utf-8')
                cur_shard_file.write(line)
                processed += 1
                num_lines_in_cur_shard += 1
    if cur_shard_file:
        print(f'#{cur_shard_id}: {num_lines_in_cur_shard} lines')
        cur_shard_file.close()


if __name__ == '__main__':
    main()
