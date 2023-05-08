from setuptools import setup, find_packages

setup(
    name="pPXF_wrapper",
    description="Wrapper around ppxf to use on all sorts of spectra",
    license="MIT License",
    author="katja",
    author_email="katja.fahrion@gmail.com",
    version="1.0.0",
    packages=find_packages(),
    install_requires=['ppxf', 'astropy'],
    entry_points={
        'console_scripts': ['ppxf_wrapper=pPXF_wrapper.ppxf_wrapper:main'],
    }
)
