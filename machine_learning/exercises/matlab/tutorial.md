Programming Exercise Tutorials
---
This post contains links to all of the programming exercise tutorials.

After clicking on a link, you may need to scroll down to find the highlighted post.

--- Note: Additional test cases can be found [(here)](https://www.coursera.org/learn/machine-learning/discussions/0SxufTSrEeWPACIACw4G5w) ---

--- Note: [(here)](https://www.coursera.org/learn/machine-learning/programming/Im1UC/regularized-linear-regression-and-bias-variance/discussions/XHpIlWbwEeW1ShIhri4C5Q/replies/s_AnhmbxEeWfZAo8m5pcCw/comments/poYJ0Wb5EeWC5xInmpgAVw) is the short story on patches.

-------------

**ex1**

[computeCost()](https://www.coursera.org/learn/machine-learning/discussions/t35D1xn3EeWA7CIAC5WDNQ) tutorial - also applies to computeCostMulti().

[gradientDescent()](https://www.coursera.org/learn/machine-learning/discussions/-m2ng_KQEeSUBCIAC9QURQ) - also applies to gradientDescentMulti() - includes test cases.

[featureNormalize()](https://www.coursera.org/learn/machine-learning/discussions/VBS-KfMxEeSUBCIAC9QURQ/replies/pRNnxfNGEeSUBCIAC9QURQ) tutorial

-------------

**ex2**

Note: If you are using MATLAB version R2015a or later, the fminunc() function has been changed in this version. The function works better, but does not give the expected result for Figure 5 in ex2.pdf, and it throws some warning messages (about a local minimum) when you run ex2_reg.m. This is normal, and you should still be able to submit your work to the grader.

Note: If your installation has trouble with the GradObj option, see this thread: [&lt;link&gt;](https://www.coursera.org/learn/machine-learning/discussions/s6tSSB9CEeWd3iIAC7VAtA)

Note: If you are using a linux-derived operating system, you may need to remove the attribute "MarkerFaceColor" from the plot() function call in plotData.m.

------------------------

[sigmoid()](https://www.coursera.org/learn/machine-learning/discussions/CA3dqwemEeWyEyIAC7PmUA/replies/sM6ONwe1EeW6KCIACxUSlA/comments/WPdF0kn6EeWBRg5dvElQww) tutorial

[costFunction()](https://www.coursera.org/learn/machine-learning/module/mgpv7/discussions/0DKoqvTgEeS16yIACyoj1Q) cost tutorial - also good for costFunctionReg()

[costFunction()](https://www.coursera.org/learn/machine-learning/discussions/GVdQ9vTdEeSUBCIAC9QURQ) gradient tutorial - also good for costFunctionReg()

[predict()](https://www.coursera.org/learn/machine-learning/discussions/_KeF3dwyEeSOoiIAC5MK1A/replies/p1gMb_6tEeSkXCIAC4tJTg) - tutorial for logistic regression prediction

Discussion of plotDecisionBoundary() [&lt;link&gt;](https://www.coursera.org/learn/machine-learning/module/mgpv7/discussions/ILWHp3g3EeWy9BLlActDnw/replies/x7zTn3hHEeWDzg4yGnIlTw/comments/DY3K83hJEeWy9BLlActDnw)

-------------

**ex3**

Note: a change to displayData.m for MacOS users: [(link)](https://www.coursera.org/learn/machine-learning/discussions/YlOmkiWsEeWeUyIAC44Ejw/replies/0A7DZi_BEeWOkCIAC4UG7w)

Note: if your images are upside, use flipud() to reverse the data - due to a change in gnuplot()'s defaults.

lrCostFunction() - note this function is identical to your costFunctionReg() from ex2.

Note: the default image orientation was changed recently in gnuplot. If your digit images are upside down, that's why. You can fix it with flipud().

[oneVsAll()](https://www.coursera.org/learn/machine-learning/discussions/8-xfvNj8EeSlUyIACycCaw/replies/KvB8afmeEeSUBCIAC9QURQ) tutorial

[predictOneVsAll()](https://www.coursera.org/learn/machine-learning/discussions/LBvBJuwsEeSQaSIACtiO2Q/replies/0WHH7OzLEeSQaSIACtiO2Q/comments/jfm5su44EeSQaSIACtiO2Q) tutorial

[predict()](https://www.coursera.org/learn/machine-learning/discussions/T-Pbwv9aEeS16yIACyoj1Q/replies/1YYwa_9dEeSUBCIAC9QURQ) tutorial (for the NN forward propagation)

-------------

**ex4**

[nnCostFunction()](https://www.coursera.org/learn/machine-learning/programming/AiHgN/neural-network-learning/discussions/QFnrpQckEeWv5yIAC00Eog) - forward propagation and cost w/ regularization

[nnCostFunction()](https://www.coursera.org/learn/machine-learning/discussions/a8Kce_WxEeS16yIACyoj1Q) - tutorial for backpropagation

[nnCostFunction()](https://www.coursera.org/learn/machine-learning/discussions/a8Kce_WxEeS16yIACyoj1Q/replies/-B1dvvW0EeS16yIACyoj1Q) - test case for backpropagation

-------------

**ex5**

[linearRegCostFunction()](https://www.coursera.org/learn/machine-learning/discussions/UAv1DB62EeWd3iIAC7VAtA) tutorial

[polyFeatures()](https://www.coursera.org/learn/machine-learning/discussions/A1WnlDJsEeWPACIACw4G5w) tutorial and discussion

[learningCurve()](https://www.coursera.org/learn/machine-learning/forum/8LDwTL2SEeSEJSIACyEKsQ/discussions/uTUjNjSqEeWPACIACw4G5w/replies/FzW6HDS1EeWddiIAC9pDDA) tips

[validationCurve()](https://www.coursera.org/learn/machine-learning/discussions/AdGhzAX1EeWyEyIAC7PmUA/replies/7XjBAQ-MEeWUtiIAC9TNkg) tips

-------------

**ex6**

Note: Update to ex6.m: At line 69/70, change "sigma = 0.5" to "sigma = %0.5f"

and change the list of output variables from "sim" to "sigma, sim".

Note: Error in visualizeBoundary.m. Change the call to contour() like this:

contour(X1, X2, vals, [1 1], 'b');

This issue can cause either the "hggroup" error message, or the decision boundaries to not be displayed, or possibly cause Octave 3.8.x to crash when running ex6.m.

[dataset3Params()](https://www.coursera.org/learn/machine-learning/module/SSmw6/discussions/zV4Url5EEeW-tBLRsFjxBw/replies/ZFgTcV7FEeWc1Q7arAoKFQ/comments/U1aoPV8YEeWvvQ62NDrKxQ) tutorial

[processEmail()](https://www.coursera.org/learn/machine-learning/programming/e4hZk/support-vector-machines/discussions/Xr_Y7BxfEeWmISIAC9QOog/replies/RayX6hySEeWmISIAC9QOog) tutorial

[emailFeatures()](https://www.coursera.org/learn/machine-learning/programming/e4hZk/support-vector-machines/discussions/unXCdEtOEeWpgBJUo9Z-Uw) tutorial

-------------

**ex7**

[findClosestCentroids()](https://www.coursera.org/learn/machine-learning/module/kxH2P/discussions/EutM1h7mEeWHGCIACznL2w/replies/7U0OvjvzEeWeYyIAC1WQXA/comments/DNu-hTxhEeWeYyIAC1WQXA) tutorial

[computeCentroids()](https://www.coursera.org/learn/machine-learning/programming/ZZkM2/k-means-clustering-and-pca/discussions/_9mZd-eDEeSnBSIAC4s0zg/replies/qaX8_ueNEeSnBSIAC4s0zg/comments/O6104jzHEeWeYyIAC1WQXA) tutorial

[Tutorials for ex7_pca functions](https://www.coursera.org/learn/machine-learning/programming/ZZkM2/k-means-clustering-and-pca/discussions/wp_NfU55EeWxHxIGetKceQ) - pca(), projectData(), recoverData()

-------------

**ex8**

- Note: error in ex8_cofi.m [(click this link)](https://www.coursera.org/learn/machine-learning/discussions/YD0v9TL_EeWj5iIACwIAYw)

[cofiCostFunc()](https://www.coursera.org/learn/machine-learning/module/HjnB4/discussions/92NKXCLBEeWM2iIAC0KUpw) tutorial