import sys
import os

# Add PaddleOCR to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paddle

def test_gpu():
    print("=" * 60)
    print("PADDLEOCR GPU CONFIGURATION TEST")
    print("=" * 60)

    # Check PaddlePaddle version
    print(f"PaddlePaddle Version: {paddle.__version__}")

    # Check CUDA compilation
    cuda_available = paddle.is_compiled_with_cuda()
    print(f"CUDA Compiled: {cuda_available}")

    if cuda_available:
        # Check GPU count
        gpu_count = paddle.device.cuda.device_count()
        print(f"GPU Count: {gpu_count}")

        if gpu_count > 0:
            # Set device to GPU
            paddle.device.set_device('gpu')
            print(f"Current Device: {paddle.get_device()}")

            # Test simple GPU operation
            print("\nTesting GPU Operation:")
            x = paddle.randn([2, 3])
            y = paddle.randn([2, 3])
            z = paddle.add(x, y)
            print(f"Simple GPU tensor operation successful!")
            print(f"Result shape: {z.shape}")

            # Check GPU memory
            print(f"\nGPU Memory Info:")
            try:
                mem_info = paddle.device.cuda.max_memory_allocated()
                print(f"Max Memory Allocated: {mem_info / 1024 / 1024:.2f} MB")
            except:
                print("Unable to get memory info")

            print("\n✅ GPU IS READY FOR TRAINING!")
            print("You can now run: python tools/train.py -c configs/rec/thai_lpr/thai_lpr_rec.yml")

            return True

    print("\n❌ GPU IS NOT AVAILABLE")
    print("The installed PaddlePaddle does not support GPU.")
    return False

if __name__ == "__main__":
    test_gpu()