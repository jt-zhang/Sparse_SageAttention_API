"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from setuptools import setup

setup(
    name="sparse_sageattn",
    version="0.1.0",
    packages=["sparse_sageattn"],  
    package_dir={"sparse_sageattn": "sparse_sageattn"},  
    package_data={
        "sparse_sageattn": ["*.py"],  # 
    },
)