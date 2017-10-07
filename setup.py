from setuptools import setup


setup(
    name='pick-em-picker',
    py_modules=['pick-em-picker'],
    install_requires=[
        'lxml>=3.8.0',
        'requests>=2.18.4',
        'toolz>=0.8.2',
        'docopt>=0.6.2',
        'typed-ast>=1.0.4'
    ],
    version='0.0.0',
    description='a pro football pick-em-picker',
    author='Adam T Johnston',
    author_email='atomjohnston@gmail.com',
    url='https://github.com/atomjohnston/pick-em-picker',
    classifiers=[
        'Programming Language :: Python',
        'Programming Langauge :: Python :: 3',
        'License :: Other/Proprietary License'
    ],
    entry_points={
        'console_scripts': [
            'pick-em=picker:main'
        ]
    }
)
