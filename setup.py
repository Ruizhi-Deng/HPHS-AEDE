import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'hphs_aede'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'param'), glob('param/*')),
        (os.path.join('share', package_name, 'world'), glob('world/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@todo.todo',
    description='HPHS-AEDE exploration algorithm (ROS2 Humble)',
    license='TODO',
    entry_points={
        'console_scripts': [
            'explorer = hphs_aede.explorer:main',
        ],
    },
)
