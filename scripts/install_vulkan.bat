@echo off
set CMAKE_ARGS=-DGGML_VULKAN=on
pip install llama-cpp-python --force-reinstall --no-cache-dir
