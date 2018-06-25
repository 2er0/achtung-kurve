from setuptools import setup, find_packages

setup(name='achtungkurve',
      version='0.1',
      description='Multiplayer Tron AI Agent playground',
      url='https://github.com/2er0/achtung-kurve',
      license='MIT',
      packages=find_packages(),
      install_requires=["numpy"],
	  extras_require={
		'gpclient': ['deap', 'dill']
	  },
      zip_safe=False
      )
