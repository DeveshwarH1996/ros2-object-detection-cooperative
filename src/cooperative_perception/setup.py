from setuptools import setup

package_name = "cooperative_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/msg", ["msg/EncodedFeature.msg"]),
        ("share/" + package_name + "/launch", ["launch/cooperative_perception.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="OpenCOOD",
    maintainer_email="placeholder@example.com",
    description="Cooperative V2X perception encoder and fusion nodes for ROS 2.",
    license="Apache-2.0",
    # tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "encoder_node = cooperative_perception.encoder_node:main",
            "fusion_node = cooperative_perception.fusion_node:main",
        ],
    },
)

