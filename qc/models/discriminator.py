import flax.linen as nn
import jax.numpy as jnp

class SuccessDiscriminator(nn.Module):
    hidden_dim: int = 512 # Tăng lên 512 hoặc 1024
    
    @nn.compact
    def __call__(self, obs, act, deterministic=True):
        x = jnp.concatenate([obs, act], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.swish(x) # Swish hoặc GELU thường mượt hơn ReLU trong RL
        x = nn.Dropout(rate=0.1, deterministic=deterministic)(x) # Tránh overfit
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.swish(x)
        x = nn.Dropout(rate=0.1, deterministic=deterministic)(x)
        x = nn.Dense(1)(x)
        return nn.sigmoid(x)