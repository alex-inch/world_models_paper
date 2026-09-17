import jax.numpy as jnp
from flax import nnx

from dreamer.wm import Decoder, DreamerConfig, Encoder

CFG = DreamerConfig()


def test_encoder_shape():
    rngs = nnx.Rngs(0)
    enc = Encoder(CFG, rngs=rngs)

    batch_size, seq_length = 5, 6
    img = rngs.normal(
        shape=(batch_size, seq_length, CFG.img_height, CFG.img_width, CFG.img_channels)
    )

    assert enc(img).shape == (batch_size, seq_length, 1536)


def test_decoder_shape():
    rngs = nnx.Rngs(0)

    enc = Decoder(CFG, rngs=rngs)

    batch_size, seq_length = 5, 6
    hidden = rngs.normal(shape=(batch_size, seq_length, CFG.wm_hidden_dim))
    latent = rngs.normal(shape=(batch_size, seq_length, CFG.latent_dim))

    joint = jnp.concatenate([hidden, latent], axis=-1)
    img = enc(joint)

    assert img.shape == (batch_size, seq_length, 64, 64, CFG.img_channels)
