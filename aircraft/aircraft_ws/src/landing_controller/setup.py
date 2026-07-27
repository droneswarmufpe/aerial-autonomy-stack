from setuptools import setup

package_name = 'landing_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/landing_controller_params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='PedroCGR',
    author_email='pcgr@cin.ufpe.br',
    maintainer='PedroCGR',
    maintainer_email='pcgr@cin.ufpe.br',
    keywords=['landing', 'tracking', 'drone', 'controller'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [
            'landing_controller = landing_controller.landing_controller_node:main',
        ],
    },
)
