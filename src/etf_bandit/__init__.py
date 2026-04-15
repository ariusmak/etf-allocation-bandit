"""etf_bandit: reusable components extracted from the research notebooks.

The notebooks remain the primary orchestration / presentation layer. This
package exposes the stable, pure-function parts so they can be reused,
tested, and imported from either scripts or the notebooks themselves.
"""

from . import config, paths  # noqa: F401
