from setuptools import setup

setup(
    name="sparse_sageattn",
    version="0.1.0",
    packages=["sparse_sageattn"],  # 主包
    package_dir={"sparse_sageattn": "sparse_sageattn"},  # 包目录位置
    # 显式声明包内的模块（确保所有文件被包含）
    package_data={
        "sparse_sageattn": ["*.py"],  # 包含所有.py文件
        # 如果需要非Python文件（如.json/.so）：
        # "sparse_sageatin": ["*.py", "data/*.json"],
    },
    # 其他配置...
)