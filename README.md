# Fast Folding Algorithm

Various implementations of the Fast-Folding Algorithm proposed by Staelin (1969). See also Kondratiev (2009). 
The FFA will fold and sum elements from an array on various periods that are determined by the algorithm. Like the FFT,
the FFA eliminates redundant summations through a clever bookkeeping of intermediate results. 

## Installation

```
python setup.py build
python setup.py install
python FFA/FFA_tests.py 
```

Also have a look at the docstrings here: 

- `FFA/FFA.py` - python implementation 
- `FFA/FFA_cy.pyx` - pure cython implementation 
- `FFA/FFA_cext.pyx` - cython extension that wraps native C code (in FFA/FFA.c)

# Contributors
- petigura 
- piscisvolans (Python 3) support
- dmopalmer 

# Wish list
- Proper unit tests
- Continuous integration 
