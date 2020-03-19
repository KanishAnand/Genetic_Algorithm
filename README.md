1. I tried to add noise using np.random.normal(0,1,11) function but due to that error value became too large.
2. There is large variation in error on running same code.

3. Right now I am only cosidering validation error in fit function. We can take effect of both training and val error.
4. Also see how to handle large variation in answers maybe through fixing seed in random

5. play with train , val ratios, mutation constants , number of children elements to mutate(currently 9)
