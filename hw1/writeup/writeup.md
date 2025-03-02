### 1.1.1

1. Gaussian: it averages out the image so that a single pixels can capture the feature of its surroundings
2. Laplacian of Gaussian: edges
3. derivative of Gaussian in the $x$ direction: vertical edges 
4. derivative of Gaussian in the $y$ direction: horizontal edges

We use multiple scales because we want to capture different sizes of surrounding response. For example, a large sigma may only capture features of neighbours one or two pixels away, while a small sigma may capture wider and large-scale features.

q2.6
alpha = 200
K = 150
[[13.  0.  0.  0.  2.  2.  1.  2.]
 [ 0. 12.  1.  2.  0.  1.  1.  3.]
 [ 1.  2.  7.  6.  1.  1.  0.  2.]
 [ 0.  2.  1. 15.  0.  0.  0.  2.]
 [ 3.  0.  1.  0. 11.  3.  0.  2.]
 [ 1.  0.  0.  1.  8. 10.  0.  0.]
 [ 2.  0.  0.  0.  0.  2. 16.  0.]
 [ 0.  1.  0.  5.  0.  0.  0. 14.]]
0.6125

[[19.  0.  0.  0.  1.  0.  0.  0.]
 [ 1. 16.  1.  0.  0.  0.  1.  1.]
 [ 0.  0. 19.  1.  0.  0.  0.  0.]
 [ 0.  0.  0. 20.  0.  0.  0.  0.]
 [ 0.  0.  0.  0. 19.  1.  0.  0.]
 [ 0.  0.  0.  0.  1. 19.  0.  0.]
 [ 0.  0.  1.  0.  0.  0. 19.  0.]
 [ 0.  0.  0.  1.  0.  0.  0. 19.]]
0.9375