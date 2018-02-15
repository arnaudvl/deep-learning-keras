
from setuptools import setup
from setuptools import find_packages

setup(name='dlkeras',
      version='1.0.1',
      description='Create, train and make predictions with CNN and feedforward NN models using keras.',
      author='Arnaud Van Looveren',
      author_email='arnaudvlooveren@gmail.com',
      url='https://github.com/arnaudvl/deep-learning-keras',
      license='MIT',
      install_requires=['keras','numpy','pandas','scikit-learn','scipy'],
      packages=find_packages())
