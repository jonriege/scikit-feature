from distutils.core import setup

NAME = "skfeature"

DESCRIPTION = "Feature Selection Repository in Python (DMML Lab@ASU)"

KEYWORDS = "Feature Selection Repository"

AUTHOR = "Jundong Li, Kewei Cheng, Suhang Wang"

AUTHOR_EMAIL = "jundong.li@asu.edu, kcheng18@asu.edu, suhang.wang@asu.edu"

URL = "https://github.com/jundongl/scikit-feature"

VERSION = "1.0.0"


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    keywords=KEYWORDS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=['skfeature', 'skfeature.utility', 'skfeature.function', 'skfeature.function.information_theory',
              'skfeature.function.similarity', 'skfeature.function.sparse_learning', 'skfeature.function.statistical',
              'skfeature.function.streaming', 'skfeature.function.structure', 'skfeature.function.wrapper',
              'skfeature.function.random_forest'],
)
