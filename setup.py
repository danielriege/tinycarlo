import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='tinycarlo',  
     version='1.0',
     author="Daniel Riege, Markus Kasten",
     description="2D car simulation",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/danielriege/tinycarlo",
     packages=setuptools.find_packages(),
     install_requires=[
        'gym>=0.21.0',
        'numpy>=1.22.0',
        'opencv-python>=4.5.5.62',
        'setuptools>=60.3.1'
     ]
 )