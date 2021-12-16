from setuptools import setup

setup(
    name='lsh-mongodb',
    version='1.0',
    description='A module to store locality sensitive hashed items into a mongodb.',
    author='kin.poon',
    author_email='ckpoon19890129@gmail.com',
    packages=['tensorflow', 'tensorflow-text',
              'tensorflow-hub', 'numpy', 'mongoengine', 'fastApi']
)
