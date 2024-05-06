from setuptools import find_packages, setup

package_name = 'tb3_controller_sixmodels'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TeemuVoutilainen',
    maintainer_email='voutilainen.teemu@gmail.com',
    description='tb3 reinforcement learning controller',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'start = tb3_controller_sixmodels.robot_controller:main',
        ],
    },
)
