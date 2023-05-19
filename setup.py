from setuptools import setup, find_packages

packages = find_packages(
    include=['quantizing', 'quantizing.*']
)

setup(name='quantizing',
      version='0.1.0',
      packages=packages
      )