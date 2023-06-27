from setuptools import setup,find_packages
from typing import List   

def get_requiremnets() -> List[str]:
    file = open("requirements.txt","r")
  

    req_list:List[str] = []
    
    return req_list
    """to install for custom code and generate libraries we should write  this setup.py file """

setup(
    name="visibility_climate",
    version="0.0.1",
    author="saikumar",
    author_email="sai680513@gmail.com",
    packages = find_packages()
    
)