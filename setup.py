from setuptools import setup

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(name='dac',
      packages=['dac', 'dac.envs'],
      platforms=['Linux', 'OSX'],
      version='0.0.1',
      install_requires=reqs,
      license='GPL-3.0',
      author=['Andre Biedenkapp'],
      author_email='biedenka@cs.uni-freiburg.de',
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Topic :: Utilities",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
      ],
      )
