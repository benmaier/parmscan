import bfmplot as bp
from bfmplot import pl
import matplotlib as mpl
import colorsys
import numpy as np

def update_min_max(minmax,vals):
    """Find new data ranges with updated data"""
    if minmax[0] is None:
        minmax[0] = np.nanmin(vals)
    else:
        if minmax[0] > np.nanmin(vals):
            minmax[0] = np.nanmin(vals)
    if minmax[1] is None:
        minmax[1] = np.nanmax(vals)
    else:
        if minmax[1] < np.nanmax(vals):
            minmax[1] = np.nanmax(vals)

    return minmax

def get_color_iterator(what_to_iterate,existing_styles=None,colors=None):

    if existing_styles is None:
        existing_styles = [ {} for _ in what_to_iterate]
    if colors is None:
        prop_cycle = mpl.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    for i, _ in enumerate(what_to_iterate):
        existing_styles[i]['color'] = colors[i]
    return existing_styles

def add_style(what_to_iterate,style_iterator):
    for focus, style in zip(what_to_iterate, style_iterator):
        k = list(focus.keys())[0]
        focus[k].update(style)
    return what_to_iterate


def add_colors(what_to_iterate,colors=None):
    styles = get_color_iterator(what_to_iterate,colors)
    return add_style(what_to_iterate, styles)


def get_linestyle_iterator(what_to_iterate,existing_styles=None,linestyles=None):

    if existing_styles is None:
        existing_styles = [ {} for _ in what_to_iterate]
    if linestyles is None:
        linestyles = ['-','-.','--',':']
    linestyles = bp.simple_cycler(linestyles)
    for i, _ in enumerate(what_to_iterate):
        existing_styles[i]['ls'] = linestyles[i]
    return existing_styles

def add_linestyles(what_to_iterate,colors=None):
    styles = get_linestyle_iterator(what_to_iterate,colors)
    return add_style(what_to_iterate, styles)


def get_linewidth_iterator(what_to_iterate,existing_styles=None,linewidths=None):

    if existing_styles is None:
        existing_styles = [ {} for _ in what_to_iterate]
    if linewidths is None:
        linewidths = [1.5,1.0,0.5]
    linewidths = bp.simple_cycler(linewidths)
    for i, _ in enumerate(what_to_iterate):
        existing_styles[i]['ls'] = linestyles[i]
    return existing_styles

def add_linewidths(what_to_iterate,colors=None):
    styles = get_linewidth_iterator(what_to_iterate,colors)
    return add_style(what_to_iterate, styles)

def get_marker_iterator(what_to_iterate,existing_styles=None,markers=None):

    if existing_styles is None:
        existing_styles = [ {} for _ in what_to_iterate]
    if markers is None:
        markers = bp.markers.items
    markers = bp.simple_cycler(markers)
    for i, _ in enumerate(what_to_iterate):
        existing_styles[i]['marker'] = markers[i]
    return existing_styles

def add_markers(what_to_iterate,colors=None):
    styles = get_marker_iterator(what_to_iterate,colors)
    return add_style(what_to_iterate, styles)

def rgb_to_hex(rgb):
    r, g, b = rgb
    _hex = '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
    return _hex

def make_lighter(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    dl = 1-l
    # manipulate h, l, s values and return as rgb
    return rgb_to_hex(
                colorsys.hls_to_rgb(
                    h,
                    min(1, l + dl * scale_l),
                    s=s,
                )
            )
