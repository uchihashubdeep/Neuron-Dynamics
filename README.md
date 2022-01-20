# Neuron-Dynamics
The Small_world Code is run on Google Collab taking advantage of CUDA GPU. There are three sections to the code. First we set the run time with necessary libraries
and then we install an extension to interface a python runtime (Colab) with the nvcc compiler.
The final part of the code is the C code which generates the small world network,remove synapse and measure the coherence. The C code will output 6 files. We will
be needing 3 of those files, times1.txt(records the spike time of each neuron),size.txt(records how many neurons fired at that instant) and index.txt(records which
neuron fired at that instant).
These three files will be used to calculate the cluster index using the final.py code . The code plots cluster index vs plots for evry synapse removed. It also plots the final cluster index value vs number of synapse removed.
The collab code can be found in the link given.
https://colab.research.google.com/drive/1D1Mjb1svZykyrFGeCsTQekVlfMTsrqko?usp=sharing
