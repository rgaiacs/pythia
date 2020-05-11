import setuptools

setuptools.setup(
    name="pythia",
    version="0.0.1",
    author="Raniere Silva",
    author_email="raniere@rgaiacs.com",
    description="Classifier for CRIC",
    long_description="",
    url="https://github.com/rgaiacs/pythia",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    scripts=['bin/pythia'],
)
