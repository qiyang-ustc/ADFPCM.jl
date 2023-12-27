import jax.numpy as jnp

class CTMState:
  def __init__(self, chi, d, D, dtype=jnp.float64):
    self.chi = chi
    self.d = d
    self.D = D
    self.dtype = dtype
    
    