# Deep Learning Test
Implementing deep learning in Rust using just a linear algebra library ([nalgebra](https://nalgebra.org/)).
The neural network (4 hidden layers, 32 neurons per layer) attempts to reconstruct a 512x512 image. It takes a 2D position as input, and outputs an RGB value.

# Results
Training time is around one minute, which is suprisingly quick considering that no parallelism or GPU acceleration is involved. Here is the reference image:

![Reference photo](https://github.com/dzamkov/deep-learning-test/blob/master/data/photo.jpg)

and here is the reconstruction from the neural network:

![Reconstruction](https://github.com/dzamkov/deep-learning-test/blob/master/reconstruct-fourier.png)

Note that I implemented Fourier features, as described [here](https://arxiv.org/pdf/2006.10739.pdf) to improve reconstruction quality. Without this, here's what the reconstruction looks like:

![Reconstruction without Fourier features](https://github.com/dzamkov/deep-learning-test/blob/master/reconstruct.png)

# Notes
* An optimizer is critical for deep learning. Otherwise, the earlier layers barely get trained at all. I used Adam Optimization, as described [here](https://www.math.purdue.edu/~nwinovic/deep_learning_optimization.html).
* Remember to reserve one column in each weight matrix for [bias](https://stackoverflow.com/questions/2480650/what-is-the-role-of-the-bias-in-neural-networks).
  This modestly improves training time and reconstruction quality.
