from setuptools import find_packages, setup

package_name = 'yolo_inference_py'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='JacopoPan',
    maintainer_email='jacopo.pan@gmail.com',
    description='YOLO inference and ROS2 publisher, OpenCV and GStreamer based',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_inference_py = yolo_inference_py.yolo_inference_node:main',
        ],
    },
)
