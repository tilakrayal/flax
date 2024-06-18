# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import typing as tp

from flax.nnx.nnx import graph
from flax.nnx.nnx.proxy_caller import CallableProxy, DelayedAccessor

A = tp.TypeVar('A')
C = tp.TypeVar('C', covariant=True)


class PureCall(tp.Protocol, tp.Generic[C]):
  def __getattr__(self, __name) -> PureCall[C]: ...

  def __getitem__(self, __name) -> PureCall[C]: ...

  def __call__(self, *args, **kwargs) -> tuple[tp.Any, C]: ...

if tp.TYPE_CHECKING:

  class PureBase(tuple[graph.GraphDef[A], graph.GraphState]):
    def __new__(
      cls,
      graphdef: graph.GraphDef[A],
      state: graph.GraphState,
    ) -> Pure[A]: ...

    nnx__pure_graphdef: graph.GraphDef[A]
    nnx__pure_state: graph.GraphState

else:

  class PureBase(tp.NamedTuple):
    nnx__pure_graphdef: graph.GraphDef
    nnx__pure_state: graph.GraphState


class Pure(PureBase[A]):
  """A Pure pytree representation of a node.

  ``Pure`` is NamedTuple containing a ``GraphDef`` and a ``State``
  which together represent a node's state. Pure can ``call`` a method
  on the underlaying node and return the result along with a new Pure
  containing the updated state.

  Example::

    >>> from flax import nnx
    >>> import jax
    >>> import jax.numpy as jnp
    ...
    >>> class StatefulLinear(nnx.Module):
    ...   def __init__(self, din, dout, rngs):
    ...     self.w = nnx.Param(jax.random.uniform(rngs(), (din, dout)))
    ...     self.b = nnx.Param(jnp.zeros((dout,)))
    ...     self.count = nnx.Variable(jnp.array(0, dtype=jnp.uint32))
    ...
    ...   def increment(self):
    ...     self.count.value += 1
    ...
    ...   def __call__(self, x):
    ...     self.increment()
    ...     return x @ self.w + self.b
    ...
    >>> linear = StatefulLinear(3, 2, nnx.Rngs(0))
    >>> pure_linear: nnx.Pure[StatefulLinear] = nnx.split(linear)
    ...
    >>> @jax.jit
    ... def forward(x, pure_linear):
    ...   y, pure_linear = pure_linear.call(x)
    ...   return y, pure_linear
    ...
    >>> x = jnp.ones((1, 3))
    >>> y, pure_linear = forward(x, pure_linear)
    >>> y, pure_linear = forward(x, pure_linear)
    ...
    >>> linear = nnx.merge(*pure_linear)
    >>> linear.count.value
    Array(2, dtype=uint32)

  In this example the ``__call__`` method was used but in general any method
  can be called. If the desired method is in a subnode, attribute access
  and indexing can be used to reach it::

    >>> rngs = nnx.Rngs(0)
    >>> nodes = dict(
    ...   a=StatefulLinear(3, 2, rngs),
    ...   b=StatefulLinear(2, 1, rngs),
    ... )
    ...
    >>> pure = nnx.split(nodes)
    >>> _, pure = pure.call['b'].increment()
    >>> nodes = nnx.merge(*pure)
    ...
    >>> nodes['a'].count.value
    Array(0, dtype=uint32)
    >>> nodes['b'].count.value
    Array(1, dtype=uint32)
  """

  @property
  def call(self) -> PureCall[Pure[A]]:
    def pure_caller(accessor: DelayedAccessor, *args, **kwargs):
      node = graph.merge(self.nnx__pure_graphdef, self.nnx__pure_state)
      method = accessor(node)
      out = method(*args, **kwargs)
      graphdef, state = graph.split(node)
      return out, Pure(graphdef, state)

    return CallableProxy(pure_caller)  # type: ignore