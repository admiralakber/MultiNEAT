# About MultiNEAT

MultiNEAT is a portable software library for performing neuroevolution, a form
of machine learning that trains neural networks with a genetic algorithm. It is
based on NEAT, an advanced method for evolving neural networks through
complexification. The neural networks in NEAT begin evolution with very simple
genomes which grow over successive generations. The individuals in the evolving
population are grouped by similarity into species, and each of them can compete
only with the individuals in the same species.

The combined effect of speciation, starting from the simplest initial structure
and the correct matching of the genomes through marking genes with historical
markings yields an algorithm which is proven to be very effective in many
domains and benchmarks against other methods.

NEAT was developed around 2002 by Kenneth Stanley in the University of Texas at
Austin.

## This fork gives you more C++ and less Python

The [Original Repository](https://github.com/peter-ch/MultiNEAT) makes things
uncessarily complicated by adding a whole heap of stuff to provide a Python
interface to the the already very expressive C++ code. This fork strips back on
all the Python and focuses on providing a nice Modern C++ library.

### License

GNU Lesser General Public License v3.0 

### Documentation
[MultiNEAT official website](http://multineat.com/docs.html)

#### To compile

To build we use [EZMake](https://github.com/thaum0/ezmake) to generate CMake
files and CMake to generate Makefiles.

EZMake generates a basic Makefile to get started that we will use below.

First, pull the git submodules:
```bash
make gitreqs
```

Next, make the build environment:
```bash
make build
```
CMake has now done its job.

Now, actually build:
```bash
cd build
make -j `nproc`
```

(Optional) Generate Doxygen documentation
```bash
make doc
xdg-open docs/html/index.html
```
