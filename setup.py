import io

from setuptools import setup, find_packages
from so_classifier import __version__

with io.open('requirements.txt', mode="r", encoding="utf-16") as f:
    required_deps = f.read().splitlines()

setup(
    name='so-classifier',
    version=__version__,
    description='so_classifier: Tag classifier for StackOverflow titles',
    url='https://github.com/btjiong/remla-extension-project',
    author='Péter Angeli, Wessel Oosterbroek, Bailey Tjiong, and Christiaan Wiers',
    keywords='machine learaning tag inference prediction similarity learning python source code stackoverflow',
    packages=find_packages(),
    python_requries='>=3.6',
    install_requires=required_deps,
    # entry_points={
    #     'console_scripts': [
    #         'so_classifier = so_classifier.train_model:main',
    #     ],
    # }
)