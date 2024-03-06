from setuptools import setup, find_packages

setup(
    name='mini-gnomix',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'typer[all]',
    ],
    entry_points={
        'console_scripts': [
            'mini-gnomix=mini_gnomix.main:app',
        ],
    },
)