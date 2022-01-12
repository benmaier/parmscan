# -*- coding: utf-8 -*-
"""
Initializes this package with metadata.
"""

from .metadata import (
        __version__,
        __author__,
        __copyright__,
        __credits__,
        __license__,
        __maintainer__,
        __email__,
        __status__,
    )

from .tools import (
        get_color_iterator,
        get_linestyle_iterator,
        get_linewidth_iterator,
        get_marker_iterator,
    )

from .parameter_plotter import ScanPlotter
