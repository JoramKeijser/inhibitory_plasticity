import numpy as onp
from jax import numpy as np
from jax import random, vmap, jit
from jax.lax import scan
from jax.numpy import clip

def split_keys(key, n_keys):
    new_keys = random.split(key, n_keys+1)
    return (k for k in new_keys[1:])

dt = 1.0
params = {'mu': -2, 'tau': 20.0, 'sigma': 1.0} # dt in ms

def rate_step(params, yt, xt):
    """
    One step of OU process
    """
    dy = (params['mu'] - yt)/params['tau'] * dt + params['sigma'] * xt * np.sqrt(dt) 
    yt = yt + dy
    return yt, yt

def ou_process(params, time_steps, neurons, key):
    y0 = np.zeros((1, ))
    x = random.normal(key, shape=(time_steps, ))
    _, y = scan(lambda y, x: rate_step(params, y, x), y0, x)
    #y = index_min(y, y<0.0, 0.0)
    #y.at[y<0.0].set(0.0)
    y = np.clip(y, 0.0, y.max()) # Non-negative
    y = y * 0.05 / y.max() #  scale to [0, 1.0]
    y = onp.squeeze(y)
    spikes = random.bernoulli(key, p = y, shape = (neurons,) + y.shape) #(neurons, groups, time)
    return x, y, spikes
    
ou_process_jit = jit(vmap(ou_process, (None, None, None, 0)), (0,1,2))