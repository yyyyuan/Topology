Command to run

%%shell

nvcc -O3 -arch=sm_75 -rdc=true \
hypercube.cu debugging.cu kernel.cu input_node.cu loading.cu vertex.cu hypercube_classifier.cu \
-o hypercube \
-lnvjpeg

./hypercube
