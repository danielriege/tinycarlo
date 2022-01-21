import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='tinycarlo',  
     version='0.1',
     scripts=['env.py'] ,
     author="Daniel Riege",
     description="OpenAI Gym for simple car simulation with segmented lane markings",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/tinycar-ai/tinycarlo",
     packages=setuptools.find_packages(),
 )