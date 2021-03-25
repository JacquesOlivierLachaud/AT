# A 2D implementation of Ambrosio-Tortorelli functional for interactive restoration.

It is fast implementation of the Ambrosio-Tortorelli functional based
on alternate minimization steps, where each step is optimized through
a semi-implicit scheme. It relies on OpenCV for data structures and display.

# Installation

OpenCV C++ headers and libs should have been installed.
In this repository:

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

# AT model Execution

```
# Increase beta to remove noise. Type 'c' to (re)start computations.
./at-gradient ../lena-512-n0_2.pgm
```

# AT superresolution model on two subsampled images

You have two programs: at-gradient-superres and at-gradient-superres-diagonals

The first one uses classical horizontal and vertical gradients. The
second one balances these gradients with diagonal gradients, for
better isotropy.

```
# Increase beta to remove noise. Type 'c' to (re)start computations.
./at-gradient-superres ../lena-512.pgm 5
./at-gradient-superres ../../Images/hepburn.pgm 5
./at-gradient-superres-diagonals ../../Images/hepburn.pgm 5
```

# Create noisy image
convert +noise Gaussian -attenuate 0.2 input.png output.pgm



