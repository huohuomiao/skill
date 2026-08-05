"""
获取本地 MLU 设备信息脚本
执行 cnmon 命令并保存设备信息到文件
"""
import subprocess
import json
import re
from datetime import datetime
from pathlib import Path


def get_device_info():
    """执行 cnmon 获取设备信息"""
    try:
        result = subprocess.run(
            ["cnmon"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"执行 cnmon 失败: {e}")
        return None


def parse_device_info(cnmon_output):
    """解析 cnmon 输出，提取设备信息"""
    devices = []

    # 匹配设备信息行
    # 格式: | 0     /   MLUXXX    v2.0.4 |         0000:11:00.0 | 0%                  0 |
    device_pattern = r'\|\s+(\d+)\s+/\s+(MLU\d+-\w+)\s+(v[\d.]+)\s+\|\s+([\w:\.]+)\s+\|'

    # 匹配内存使用行
    # 格式: | /         41C     70 W/ 350 W |     0 MiB/ 32768 MiB | FULL          Default |
    memory_pattern = r'\|\s+/\s+.*?\|\s+(\d+)\s+MiB/\s+(\d+)\s+MiB\s+\|'

    device_lines = cnmon_output.split('\n')
    i = 0
    while i < len(device_lines):
        line = device_lines[i]
        device_match = re.search(device_pattern, line)

        if device_match:
            card_id = device_match.group(1)
            device_name = device_match.group(2)
            firmware = device_match.group(3)
            bus_id = device_match.group(4)

 # 查找下一行的内存信息
            memory_used = 0
            memory_total = 0
            if i + 1 < len(device_lines):
                memory_match = re.search(memory_pattern, device_lines[i + 1])
                if memory_match:
                    memory_used = int(memory_match.group(1))
                    memory_total = int(memory_match.group(2))

            devices.append({
                "card_id": card_id,
                "device_name": device_name,
                "firmware": firmware,
                "bus_id": bus_id,
                "memory_used": memory_used,
                "memory_total": memory_total
            })

        i += 1

    return devices

def check_memory_status(devices):
    """检查内存占用状态，返回是否有设备内存被占用"""
    has_memory_usage = False
    for dev in devices:
        if dev['memory_used'] > 0:
            has_memory_usage = True
            print(f"\n⚠️  警告: Card {dev['card_id']} 上有内存被占用 ({dev['memory_used']} MiB)")
            print(f"   当前可能有其他测试在执行，优化得到的数据不一定准确")

    return has_memory_usage

def main():
    """主函数"""
    print("=" * 60)
    print("MLU 设备信息采集")
    print("=" * 60)

    # 执行 cnmon
    print("\n执行 cnmon 命令...")
    cnmon_output = get_device_info()

    if cnmon_output is None:
        print("✗ cnmon 命令不可用或执行失败")
        return False

    print("✓ cnmon 命令执行成功")

    # 解析设备信息
    devices = parse_device_info(cnmon_output)
    print(f"✓ 检测到 {len(devices)} 个 MLU 设备")

 print("\n设备信息:")
    for dev in devices:
        memory_usage = f"{dev['memory_used']} / {dev['memory_total']} MiB"
        print(f"  - Card {dev['card_id']}: {dev['device_name']} (Firmware: {dev['firmware']}, Bus: {dev['bus_id']}, Memory: {memory_usage})")

    # 检查内存占用
    has_memory_usage = check_memory_status(devices)

    print("\n" + "=" * 60)
    print("✓ 设备信息采集完成")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)