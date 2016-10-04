#!/usr/bin/env python
# encoding: utf-8

import os
from setuptools.command.install import install
from setuptools import setup


with open('README.rst') as f:
    LONG_DESCRIPTION = f.read()


class MakeCommand(install):
    def run(self):
        os.system('make')
        common_dir = 'yast/learner'
        target_dir = '%s/%s' % (self.build_lib, common_dir)
        self.mkpath(target_dir)
        os.system('cp %s/util.so.1 %s' % (common_dir, target_dir))
        # fuck_dir = 'yast/learner'
        common_dir ='yast/learner/liblinear'
        target_dir = '%s/%s' % (self.build_lib, common_dir)
        self.mkpath(target_dir)
        os.system('cp %s/liblinear.so.1 %s' % (common_dir, target_dir))
        install.run(self)

        # os.system('chmod 777 yast/learner/learner_impl.py') # Execute by owner.

setup(
    name='yast',
    version='0.2.7',
    packages=['yast', 'yast.learner', 'yast.learner.liblinear.python'],
    url='',
    license='BSD',
    author='jinyang',
    author_email='jinyang.zhou@guokr.com',
    description='Yet Another short text (toolkit based on LibLinear)',
    long_description=LONG_DESCRIPTION,
    install_requires=['jieba'],
    keywords='text classification svm liblinear libshorttext',
    cmdclass={'install': MakeCommand}
)
