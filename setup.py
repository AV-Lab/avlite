from setuptools import find_packages, setup
from glob import glob

package_name = 'avlite'

# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()

setup(
    name=package_name,
    version='0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['config.yaml']),
        ('share/' + package_name + '/resource', glob('resource/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mkhonji',
    maintainer_email='majid.khonji@gmail.com',
    description='AVLite - Modular Autonomous Vehicle Stack',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'avlite = avlite.__main__:main'
        ],
    },
)



