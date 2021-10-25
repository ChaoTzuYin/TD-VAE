# Temporal Difference Variational Auto-Encoder (TD-VAE)
The code implements the framework proposed in Temporal Difference Variational Auto-Encoder (Gregor et al).
The main purpose is to make it a simple block that is easy to use as a plugin of the network.
As the sampling process in the multi-stochastic-layer version is extremely complicated, I try the trick of recursive function call to simplify it.
<br/>
![plot](./figures/TDVAE.PNG)
# Result: The moving digist experiment
<br/>
Ground truth
<img src="./figures/MD_gt.gif" alt="Italian Trulli" 
 width="400" 
 height="200">
<br/>
Single state predition
<img src="./figures/MD_S2S.gif" alt="Italian Trulli"
 width="400" 
 height="200"">
<br/>
Roll out (Recursive prediction)
<img src="./figures/MD_rollout.gif" alt="Italian Trulli"
 width="400" 
 height="200">

