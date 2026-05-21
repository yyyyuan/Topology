cmd to compile manifold in Bo's local machine **with parallel options using OpenMP** on CPU:
> clang++ -std=c++17 -Xpreprocessor -fopenmp   -I/usr/local/Cellar/libomp/22.1.4/include   -L/usr/local/Cellar/libomp/22.1.4/lib   -lomp manifold.cpp -o manifold

**Use Makefile to build**

Default build, using sequentianl execution in CPU mode:
> make all

Use OpenMP to do parallel executino:
> make parallel