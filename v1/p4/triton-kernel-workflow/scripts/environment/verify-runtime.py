"""
Triton 环境检查脚本 - Vector Add 示例
用于验证 Triton 环境是否就绪
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """简单的向量加法 kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """向量加法 wrapper"""
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=128)
    return output


def main():
    """环境检查主函数"""
    print("=" * 60)
    print("Triton 环境检查 - Vector Add 示例")
    print("=" * 60)

    # 检查 Triton 版本
    print(f"\n✓ Triton 版本: {triton.__version__}")

    # 检查 PyTorch 版本
    print(f"✓ PyTorch 版本: {torch.__version__}")

    # 检查 MLU 设备
    if not torch.mlu.is_available():
        print("✗ MLU 设备不可用")
        return False
    print(f"✓ MLU 设备可用")

    # 简单的功能测试
    print("\n执行 Vector Add 测试...")
    try:
        device = torch.device("mlu")

        # 创建测试数据
        size = 1024
        x = torch.randn(size, device=device, dtype=torch.float32)
        y = torch.randn(size, device=device, dtype=torch.float32)

        # 执行 Triton kernel
        output = add(x, y)

        # 验证精度
        expected = x + y
        diff = torch.abs(output - expected).max().item()

        print(f"✓ Vector Add 执行成功")
        print(f"✓ 最大误差: {diff:.2e}")

        # 精度检查
        if diff < 1e-5:
            print(f"✓ 精度验证通过 (diff < 1e-5)")
            return True
        else:
            print(f"✗ 精度验证失败 (diff = {diff:.2e})")
            return False

    except Exception as e:
        print(f"✗ 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print()
    success = main()
    print("\n" + "=" * 60)
    if success:
        print("✓ 环境就绪，可以开始 Triton 算子开发")
        print("=" * 60)
        exit(0)
    else:
        print("✗ 环境不就绪，请检查配置")
        print("=" * 60)
        exit(1)
