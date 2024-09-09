from setuptools import setup, find_packages

from llm_planner import __version__

setup(
    name="llm_planner",
    version=__version__,
    author="Data Systems Group, MPI-SWS",
    author_email="foo@bar",
    url="foo.bar",
    description="A Natural Language Message Planner",
    long_description="A Planner for Serving Natural Language Requests",
    packages=find_packages(exclude=("experiments", "scripts", "third_party"),
                           include=['llm_planner', 'llm_planner.*']),
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=[],
    #install_requires=['setuptools==69.5.1', 'yapf==0.40.2'],
)
