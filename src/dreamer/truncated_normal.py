"""Truncated normal distributions for bounded continuous actions."""

import math

import chex
import distrax
import jax
import jax.numpy as jnp
from jax.scipy.special import erfcx, log_ndtr, ndtr, ndtri

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
_SQRT_PI_OVER_2 = math.sqrt(math.pi / 2)
_SQRT_2 = math.sqrt(2)


def _ndtri_from_log_cdf(log_p):
    """Inverse normal CDF without underflowing very small probabilities."""
    ordinary = ndtri(jnp.exp(jnp.maximum(log_p, -60.0)))
    target = jnp.minimum(log_p, -60.0)
    initial = -jnp.sqrt(-2 * target)

    def newton_step(_, x):
        log_cdf = log_ndtr(x)
        slope = jnp.exp(-0.5 * x**2 - _HALF_LOG_2PI - log_cdf)
        return x - (log_cdf - target) / slope

    tail = jax.lax.fori_loop(0, 4, newton_step, initial)
    return jnp.where(log_p > -60, ordinary, tail)


def _tail_stats(a, width, scale):
    """Mean distance from the nearer bound, entropy, and log mass for a >= 0."""
    b = a + width
    ratio = jnp.exp(-0.5 * width * (a + b))  # phi(b) / phi(a)
    mills_a = _SQRT_PI_OVER_2 * erfcx(a / _SQRT_2)
    mills_b = _SQRT_PI_OVER_2 * erfcx(b / _SQRT_2)
    far = (a >= 20) & (a * width > 20)
    integral = jnp.where(far, mills_a, mills_a - ratio * mills_b)
    ordinary_displacement = -jnp.expm1(-0.5 * width * (a + b)) / integral - a
    inv_a = 1 / jnp.maximum(a, 20.0)
    log_integral = jnp.where(
        far,
        -jnp.log(jnp.maximum(a, 20.0))
        + jnp.log1p(-(inv_a**2) + 3 * inv_a**4 - 15 * inv_a**6 + 105 * inv_a**8),
        jnp.log(integral),
    )
    asymptotic_displacement = inv_a * (1 - 2 * inv_a**2 + 10 * inv_a**4 - 74 * inv_a**6)
    displacement = jnp.where(far, asymptotic_displacement, ordinary_displacement)
    entropy = (
        jnp.log(scale)
        + log_integral
        + 0.5 * (1 + a * displacement - jnp.where(far, 0.0, width * ratio / integral))
    )
    log_mass = -0.5 * a**2 - _HALF_LOG_2PI + log_integral
    return displacement, entropy, log_mass, log_integral


def _sample_far_tail(a, width, uniform):
    """Invert the survival ratio relative to a, preserving far-tail precision."""
    mills_a = _SQRT_PI_OVER_2 * erfcx(a / _SQRT_2)
    mills_b = _SQRT_PI_OVER_2 * erfcx((a + width) / _SQRT_2)
    log_remaining = -0.5 * width * (2 * a + width) + jnp.log(mills_b / mills_a)
    target = jnp.log1p(uniform * jnp.expm1(log_remaining))

    def newton_step(_, z):
        mills_z = _SQRT_PI_OVER_2 * erfcx((a + z) / _SQRT_2)
        log_ratio = -a * z - 0.5 * z**2 + jnp.log(mills_z / mills_a)
        return z + (log_ratio - target) * mills_z

    return jax.lax.fori_loop(0, 3, newton_step, -target / a)


class _ScalarTruncatedNormal(distrax.Distribution):
    """Elementwise Normal(loc, scale) conditioned on lower <= x <= upper.

    Parameters broadcast as Distrax scalar batch dimensions.
    Requires positive scale and lower < upper.
    """

    def __init__(self, loc, scale, lower, upper):
        self._normal = distrax.Normal(loc, scale)
        dtype = jnp.result_type(self._normal.loc, self._normal.scale)
        self._lower = jnp.asarray(lower, dtype=dtype)
        self._upper = jnp.asarray(upper, dtype=dtype)
        jax.lax.broadcast_shapes(
            self._normal.batch_shape, self._lower.shape, self._upper.shape
        )

    @property
    def event_shape(self):
        return ()

    @property
    def batch_shape(self):
        return jax.lax.broadcast_shapes(
            self._normal.batch_shape, self._lower.shape, self._upper.shape
        )

    @property
    def loc(self):
        return jnp.broadcast_to(self._normal.loc, self.batch_shape)

    @property
    def scale(self):
        return jnp.broadcast_to(self._normal.scale, self.batch_shape)

    @property
    def lower(self):
        return jnp.broadcast_to(self._lower, self.batch_shape)

    @property
    def upper(self):
        return jnp.broadcast_to(self._upper, self.batch_shape)

    def _statistics(self):
        loc, scale, lower, upper = self.loc, self.scale, self.lower, self.upper
        width = (upper - lower) / scale
        a = (lower - loc) / scale
        b = (upper - loc) / scale
        center = (lower + upper) / 2
        c = (center - loc) / scale
        h = width / 2

        # CDF differences and endpoint-density differences lose precision when
        # the standardized interval is short and its density is nearly flat.
        small = (h < 0.1) & (jnp.abs(c) * h < 0.1)
        hs = jnp.where(small, h, 0.05)
        cs = jnp.where(small, c, 0.0)
        h2, c2 = hs**2, cs**2
        mass = 1 + (c2 - 1) * h2 / 6 + (c2**2 - 6 * c2 + 3) * h2**2 / 120
        first = (-cs * h2 / 3 + (3 * cs - cs**3) * h2**2 / 30) / mass
        second = (h2 / 3 + (c2 - 1) * h2**2 / 10) / mass
        small_mean = center + scale * first
        small_entropy = jnp.log(upper - lower) + jnp.log(mass) + cs * first + second / 2
        small_log_mass = jnp.log(2 * hs) - c2 / 2 - _HALF_LOG_2PI + jnp.log(mass)

        positive = a >= 0
        negative = b <= 0
        pos_a = jnp.where(positive & ~small, a, 0.0)
        pos_w = jnp.where(positive & ~small, width, 1.0)
        pos_dx, pos_entropy, pos_log_mass, pos_log_integral = _tail_stats(
            pos_a, pos_w, scale
        )
        neg_a = jnp.where(negative & ~small, -b, 0.0)
        neg_w = jnp.where(negative & ~small, width, 1.0)
        neg_dx, neg_entropy, neg_log_mass, neg_log_integral = _tail_stats(
            neg_a, neg_w, scale
        )

        cross = ~positive & ~negative & ~small
        cross_a = jnp.where(cross, a, -1.0)
        cross_b = jnp.where(cross, b, 1.0)
        cross_log_mass = jnp.log1p(-ndtr(cross_a) - ndtr(-cross_b))
        density_a = jnp.exp(-0.5 * cross_a**2 - _HALF_LOG_2PI - cross_log_mass)
        density_b = jnp.exp(-0.5 * cross_b**2 - _HALF_LOG_2PI - cross_log_mass)
        cross_mean = loc + scale * (density_a - density_b)
        cross_entropy = (
            jnp.log(scale)
            + cross_log_mass
            + _HALF_LOG_2PI
            + 0.5 * (1 + cross_a * density_a - cross_b * density_b)
        )

        mean = jnp.where(
            small,
            small_mean,
            jnp.where(
                positive,
                lower + scale * pos_dx,
                jnp.where(negative, upper - scale * neg_dx, cross_mean),
            ),
        )
        entropy = jnp.where(
            small,
            small_entropy,
            jnp.where(
                positive, pos_entropy, jnp.where(negative, neg_entropy, cross_entropy)
            ),
        )
        log_mass = jnp.where(
            small,
            small_log_mass,
            jnp.where(
                positive,
                pos_log_mass,
                jnp.where(negative, neg_log_mass, cross_log_mass),
            ),
        )
        return (
            mean,
            entropy,
            log_mass,
            (small, positive, negative, pos_log_integral, neg_log_integral, c, mass),
        )

    def mean(self):
        return self._statistics()[0]

    def entropy(self):
        return self._statistics()[1]

    def log_prob(self, value):
        value = jnp.asarray(value)
        (
            _,
            _,
            log_mass,
            (small, positive, negative, pos_log_integral, neg_log_integral, c, mass),
        ) = self._statistics()
        scale = self.scale
        standardized = (value - self.loc) / scale
        small_u = (value - (self.lower + self.upper) / 2) / scale
        small_log_prob = (
            -0.5 * small_u * (small_u + 2 * c)
            - jnp.log(self.upper - self.lower)
            - jnp.log(mass)
        )
        pos_u = (value - self.lower) / scale
        pos_a = (self.lower - self.loc) / scale
        pos_log_prob = (
            -0.5 * pos_u * (pos_u + 2 * pos_a) - jnp.log(scale) - pos_log_integral
        )
        neg_u = (self.upper - value) / scale
        neg_a = (self.loc - self.upper) / scale
        neg_log_prob = (
            -0.5 * neg_u * (neg_u + 2 * neg_a) - jnp.log(scale) - neg_log_integral
        )
        cross_log_prob = (
            -0.5 * standardized**2 - _HALF_LOG_2PI - jnp.log(scale) - log_mass
        )
        result = jnp.where(
            small,
            small_log_prob,
            jnp.where(
                positive,
                pos_log_prob,
                jnp.where(negative, neg_log_prob, cross_log_prob),
            ),
        )
        return jnp.where(
            (value >= self.lower) & (value <= self.upper), result, -jnp.inf
        )

    def _sample_n(self, key, n):
        shape = (n,) + self.batch_shape
        lower, upper, loc, scale = self.lower, self.upper, self.loc, self.scale
        a, b = (lower - loc) / scale, (upper - loc) / scale
        u = jax.random.uniform(key, shape=shape, dtype=loc.dtype)
        log_u, log_one_minus_u = jnp.log(u), jnp.log1p(-u)
        log_cdf = jnp.logaddexp(log_ndtr(a) + log_one_minus_u, log_ndtr(b) + log_u)
        log_sf = jnp.logaddexp(log_ndtr(-a) + log_one_minus_u, log_ndtr(-b) + log_u)
        standard = jnp.where(
            a >= 0, -_ndtri_from_log_cdf(log_sf), _ndtri_from_log_cdf(log_cdf)
        )
        sample = loc + scale * standard
        width = (upper - lower) / scale
        far_positive = a >= 20
        far_negative = b <= -20
        pos_a = jnp.where(far_positive, a, 20.0)
        pos_w = jnp.where(far_positive, width, 1.0)
        neg_a = jnp.where(far_negative, -b, 20.0)
        neg_w = jnp.where(far_negative, width, 1.0)
        sample = jnp.where(
            far_positive, lower + scale * _sample_far_tail(pos_a, pos_w, u), sample
        )
        sample = jnp.where(
            far_negative, upper - scale * _sample_far_tail(neg_a, neg_w, u), sample
        )
        return jnp.clip(sample, lower, upper)

    def __getitem__(self, index):
        return _ScalarTruncatedNormal(
            self.loc[index], self.scale[index], self.lower[index], self.upper[index]
        )


class TruncatedNormal(distrax.Independent):
    """Implementation of a truncated normal distrax distribution. The builtin distrax ClippedNormal
    doesn't support entropy computation - we extend it to do so.

    For parameters shaped ``[..., action_dim]``, samples and means have that
    shape, while log probabilities and entropies have shape ``[...]``.
    Requires positive scale and lower < upper in each action coordinate.
    """

    def __init__(self, loc, scale, lower, upper):
        loc_array = jnp.asarray(loc)
        chex.assert_shape(loc_array, (..., None))
        action_dim = loc_array.shape[-1]
        for parameter in (scale, lower, upper):
            array = jnp.asarray(parameter)
            if array.ndim:
                chex.assert_axis_dimension(array, -1, action_dim)
        scalar = _ScalarTruncatedNormal(loc, scale, lower, upper)
        chex.assert_equal_shape((scalar.loc, loc_array))
        super().__init__(scalar, reinterpreted_batch_ndims=1)

    @property
    def loc(self):
        return self.distribution.loc

    @property
    def scale(self):
        return self.distribution.scale

    @property
    def lower(self):
        return self.distribution.lower

    @property
    def upper(self):
        return self.distribution.upper
