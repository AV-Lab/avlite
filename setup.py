from setuptools import find_packages, setup
from glob import glob

package_name = 'race_plan_control'

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
    description='An autonomous racing stack',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'race_plan_control = race_plan_control.main:run'
        ],
    },
)



