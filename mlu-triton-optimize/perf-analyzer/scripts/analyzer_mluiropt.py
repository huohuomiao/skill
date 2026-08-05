import re
import sys
import torch
from triton.backends.mlu import driver


def extract_memref_sizes(file_path):
    pattern = re.compile(r'memref<(\d+)x[^,>]+,\s*101>')
    results = []

    with open(file_path, 'r') as f:
        for line in f:
            if 'mlu.alloc' in line:
                match = pattern.search(line)
                if match:
                    results.append(int(match.group(1)))
    
    return results




def main():
    if len(sys.argv) != 2:
        print("Usage: python extract.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    nram = extract_memref_sizes(file_path)
    
    _devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())
    _MAX_NRAM_SIZE = _devprob.get('max_nram_size')

    TOTAL_CORE_NUM = _devprob.get('cluster_num') * _devprob.get("core_num_per_cluster")

    print(f"TOTAL_CORE_NUM = {TOTAL_CORE_NUM} ")
    print(f"_MAX_NRAM_SIZE = {_MAX_NRAM_SIZE} B")
    print(f"NRAM_used = {nram[0]} B")
    print(f"Utilization: {nram[0]/_MAX_NRAM_SIZE *100} %")
if __name__ == "__main__":
    main()

