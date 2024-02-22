import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='tinycarlo',  
     version='2.0.0',
     author="Daniel Riege",
     description="Think of Carla but much lighter and simpler or the Car Racing gym but more serious",
     license="MIT",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/danielriege/tinycarlo",
     packages=["tinycarlo", "tinycarlo.wrapper"],
     install_requires=[
        'gymnasium>=0.26.0',
        'numpy>=1.22.0',
        'opencv-python>=4.5.5.62',
        'setuptools>=60.3.1',
        'pyyaml>=6.0'
     ],
     python_requires='>=3.8',
     extras_require={
        'testing': ['pytest'],    
    },
     entry_points={
        'console_scripts': [
            'tinycarlo.mapbuilder=mapbuilder.mapbuilder:main',
        ],
    },
 )