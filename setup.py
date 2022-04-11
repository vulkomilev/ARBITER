from setuptools import find_packages, setup
setup(
    name='arbiter',
    packages=find_packages(include=['arbiter']),
    version='0.1.0',
    description='The Arbiter Python library',
    author='Valko Milev',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)