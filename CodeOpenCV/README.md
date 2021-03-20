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

# Execution

```
# Increase beta to remove noise. Type 'c' to (re)start computations.
./at-gradient ../lena-512-n0_2.pgm
```

# Create noisy image
convert +noise Gaussian -attenuate 0.2 input.png output.pgm



