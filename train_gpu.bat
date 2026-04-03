@echo off
echo ========================================
echo PaddleOCR GPU Training Script
echo ========================================
echo.
echo Using Python 3.11 environment with GPU support
echo GPU: NVIDIA GeForce RTX 2060
echo CUDA: 12.4
echo.

REM Use the Python from the GPU-enabled conda environment
set PYTHON_GPU=C:\Users\User\anaconda3\envs\paddleocr_gpu\python.exe

echo Checking GPU availability...
%PYTHON_GPU% -c "import paddle; print('GPU Available:', paddle.is_compiled_with_cuda()); print('GPU Count:', paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0)"

echo.
echo Starting training with GPU...
echo Command: %PYTHON_GPU% tools/train.py -c configs/rec/thai_lpr/thai_lpr_rec.yml
echo.

%PYTHON_GPU% tools/train.py -c configs/rec/thai_lpr/thai_lpr_rec.yml

pause