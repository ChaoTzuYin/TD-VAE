# Temporal Difference Variational Auto-Encoder (TD-VAE)
<br>- The code implements the framework proposed in <a href="https://arxiv.org/abs/1806.03107">Temporal Difference Variational Auto-Encoder (Gregor et al)</a>.<br/>
<br>- The main purpose of this implementation is to make it a simple block that is easy to use as a plugin of the network.<br/>
<br>- As the sampling process in the multi-stochastic-layer version is extremely complicated, I try the trick of recursive function call to simplify it.<br/>
<br/>
![plot](./figures/TDVAE.PNG)
# Result: The moving digist experiment
<br>
The task of moving digist is mainly to predict the next digist state after observing a sequence of moving digists. Dataset used in this experiment is as defined in "moving_mnist.py". Notice for the row 6 of "moving_mnist.py", before you use it, please make sure to either change the root path of mnist dataset according to your situation or to change the "download" flag into True. The results are as shown in below.
<br/>

<br/>
<strong> Ground truth </strong>
<img src="./figures/MD_gt.gif" alt="Italian Trulli" 
 width="400" 
 height="200">
<br/>
<strong> Single state predition</strong>
<img src="./figures/MD_S2S.gif" alt="Italian Trulli"
 width="400" 
 height="200">
<br/>
<strong> Roll out (Recursive prediction)</strong>
<img src="./figures/MD_rollout.gif" alt="Italian Trulli"
 width="400" 
 height="200">
<br/>

