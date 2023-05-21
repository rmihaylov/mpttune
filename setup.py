from setuptools import setup, find_packages


setup(
    name='mpttune',
    version='0.1.0',
    packages=find_packages(include=['mpttune', 'mpttune.*']),
    entry_points={
        'console_scripts': ['mpttune=mpttune.run:main']
    }
)
