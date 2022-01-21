import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='tinycarlo',  
     version='0.1',
     author="Daniel Riege, Markus Kasten",
     description="OpenAI Gym for simple car simulation with segmented lane markings",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/tinycar-ai/tinycarlo",
     packages=setuptools.find_packages(),
     install_requires=[
        'gym>=0.21.0',
        'numpy>=1.22.0',
        'opencv-python>=4.5.5.62',
        'setuptools>=60.3.1'

     ]
 )