import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HCDSIM",
    version="1.3.0",
    author="Xikang Feng",
    author_email="fxk@nwpu.edu.cn",
    maintainer="Sisi Peng",
    maintainer_email="sisipeng@mail.nwpu.edu.cn",
    description="HCDSIM: A Single-Cell Genomics Simulator with Haplotype-Specific Copy Number Annotation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xikanfeng2/HCDSIM",
    project_urls={
        "Bug Tracker": "https://github.com/xikanfeng2/HCDSIM/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords=[
        'cancer',
        'single-cell',
        'DNA',
        'copy-number',
        'haplotype-specific',
        'simulator',
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'pandas',
        'numpy>=1.16.1,<1.25',
        'matplotlib>=3.0.2',
        'networkx>=3.2.1',
        'scikit-learn',
        'biopython',
    ],
    entry_points={
        'console_scripts': [
            'hcdsim=hcdsim.bin.hcdsim_main:main',
            'hcdbench=hcdsim.bin.hcdbench_main:main'
        ],
    },
)