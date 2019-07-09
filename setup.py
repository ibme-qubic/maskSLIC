
"""
Setup script for maskslic
"""
import os
import subprocess
import re
import io
import glob
import sys

from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension

from Cython.Build import cythonize

import numpy

MODULE = 'maskslic'
MODULE_DIR = os.path.abspath(os.path.dirname(__file__))

def get_filetext(rootdir, filename):
    """ Get the text of a local file """
    with io.open(os.path.join(rootdir, filename), encoding='utf-8') as fhandle:
        return fhandle.read()

def git_version():
    """ Get the full and python standardized version from Git tags (if possible) """
    try:
        # Full version includes the Git commit hash
        full_version = subprocess.check_output('git describe --dirty', shell=True).decode("utf-8").strip(" \n")

        # Python standardized version in form major.minor.patch.post<build>
        version_regex = re.compile(r"v?(\d+\.\d+(\.\d+)?(-\d+)?).*")
        match = version_regex.match(full_version)
        if match:
            std_version = match.group(1).replace("-", ".post")
        else:
            raise RuntimeError("Failed to parse version string %s" % full_version)
        return full_version, std_version
    except:
        # Any failure, return None. We may not be in a Git repo at all
        return None, None

def git_timestamp():
    """ Get the last commit timestamp from Git (if possible)"""
    try:
        return subprocess.check_output('git log -1 --format=%cd', shell=True).decode("utf-8").strip(" \n")
    except:
        # Any failure, return None. We may not be in a Git repo at all
        return None

def update_metadata(rootdir, version_str, timestamp_str):
    """ Update the version and timestamp metadata in the module _version.py file """
    with io.open(os.path.join(rootdir, MODULE, "_version.py"), "w", encoding='utf-8') as fhandle:
        fhandle.write("__version__ = '%s'\n" % version_str)
        fhandle.write("__timestamp__ = '%s'\n" % timestamp_str)

def get_requirements(rootdir):
    """ Get a list of all entries in the requirements file """
    with io.open(os.path.join(rootdir, 'requirements.txt'), encoding='utf-8') as fhandle:
        return [l.strip() for l in fhandle.readlines()]

def get_version(rootdir):
    """ Get the current version number (and update it in the module _version.py file if necessary)"""
    version, timestamp = git_version()[1], git_timestamp()

    if version is not None and timestamp is not None:
        # We got the metadata from Git - update the version file
        update_metadata(rootdir, version, timestamp)
    else:
        # Could not get metadata from Git - use the version file if it exists
        with io.open(os.path.join(rootdir, MODULE, '_version.py'), encoding='utf-8') as fhandle:
            metadata = fhandle.read()
            match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", metadata, re.M)
            if match:
                version = match.group(1)
            else:
                version = "unknown"
    return version

def get_package_data(_rootdir):
    """
    Get extra data files to install into the package, e.g. icons
    """
    return {
        MODULE : glob.glob("%s/*.png" % MODULE) + glob.glob("%s/*.svg" % MODULE)
    }

def get_extensions(_rootdir):
    """
    Get Cython extensions
    """
    compile_args = []
    link_args = []

    if sys.platform.startswith('win'):
        compile_args.append('/EHsc')
    elif sys.platform.startswith('darwin'):
        compile_args += ["-mmacosx-version-min=10.9"]
        link_args += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]

    #extensions.append(Extension("%s.perfusionslic.additional.bspline_smoothing" % MODULE,
    #                            sources=["%s/perfusionslic/additional/bspline_smoothing.pyx" % MODULE],
    #                            include_dirs=[numpy.get_include()]))

    #extensions.append(Extension("%s.perfusionslic.additional.create_im" % MODULE,
    #                            sources=["%s/perfusionslic/additional/create_im.pyx" % MODULE],
    #                            include_dirs=[numpy.get_include()]))

    extensions = [
        Extension("%s._slic" % MODULE,
                  sources=["%s/_slic.pyx" % MODULE],
                  include_dirs=[numpy.get_include()]),

        Extension("%s.processing" % MODULE,
                  sources=["%s/processing.pyx" % MODULE,
                           "src/processing.cpp"],
                  include_dirs=["src", numpy.get_include()],
                  language="c++",
                  extra_compile_args=compile_args, extra_link_args=link_args),
    ]

    return cythonize(extensions)

KWARGS = {
    'name' : 'maskslic',
    'version' : get_version(MODULE_DIR),
    'description' : 'Simple linear iterative clustering (SLIC) in a region of interest (ROI)',
    'long_description' : get_filetext(MODULE_DIR, 'README.md'),
    'long_description_content_type' : 'text/markdown',
    'url' : 'https://www.birving.com',
    'author' : 'Benjamin Irving',
    'author_email' : 'mail@birving.com',
    'license' : 'Copyright (C) 2016-2019, Benjamin Irving. See LICENSE file for distribution conditions',
    'setup_requires' : ['cython'],
    'install_requires' : get_requirements(MODULE_DIR),
    'packages' : find_packages(),
    'package_data' : get_package_data(MODULE_DIR),
    'ext_modules' : get_extensions(MODULE_DIR),
    'include_package_data' : True,
    'classifiers' : [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: Free for non-commercial use',
    ],
}

setup(**KWARGS)
