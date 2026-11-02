from setuptools import setup, find_packages

setup(
    name="pulmonary-artery-vein-separation",
    version="1.0.0",
    description="Enhanced Hierarchical Graph Attention Networks with Anatomical Priors for Pulmonary Artery-Vein Separation",
    author="Pulmonary Artery-Vein Separation Research Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torch-geometric>=2.4.0",
        "monai>=1.3.0",
        "nibabel>=5.1.0",
        "SimpleITK>=2.3.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0,<2.0.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.1",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "openpyxl>=3.1.0"
    ],
)

