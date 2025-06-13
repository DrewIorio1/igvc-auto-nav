from setuptools import setup, find_packages

setup(
    name='fusion_node',
    version='0.1.2',
    packages=find_packages(),  # Automatically finds packages in fusion_node/
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='University of Bridgeport',
    maintainer_email='aiorio@my.bridgeport.edu',
    description='Sensor fusion and path planning node for ROS2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'fusion_node = fusion_node.fusion_node:main',
        ],
    },
)