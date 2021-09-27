from setuptools import setup, Extension

module = Extension ('lab2', sources=['lab2.pyx'])

setup(
    name='lab2',
    version='1.0',
    author='jet',
    ext_modules=[module]
)