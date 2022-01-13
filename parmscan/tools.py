import bfmplot as bp
from bfmplot import pl
import colorsys

def update_min_max(minmax,vals):
    """Find new data ranges with updated data"""
    if minmax[0] is None:
        minmax[0] = min(vals)
    else:
        if minmax[0] > min(vals):
            minmax[0] = min(vals)
    if minmax[1] is None:
        minmax[1] = max(vals)
    else:
        if minmax[1] < max(vals):
            minmax[1] = max(vals)

    return minmax

def get_color_iterator(what_to_iterate,existing_styles=None,colors=None):

    if existing_styles is None:
        existing_styles = [ {} for _ in what_to_iterate]
    if colors is None:
        colors = bp.colors
    colors = bp.simple_cycler(colors)
    for i, _ in enumerate(what_to_iterate):
        existing_styles[i]['color'] = colors[i]
    return existing_styles

def get_linestyle_iterator(what_to_iterate,existing_styles=None,linestyles=None):

    if existing_styles is None:
        existing_styles = [ {} for _ in what_to_iterate]
    if linestyles is None:
        linestyles = ['-','-.','--',':']
    linestyles = bp.simple_cycler(linestyles)
    for i, _ in enumerate(what_to_iterate):
        existing_styles[i]['ls'] = linestyles[i]
    return existing_styles

def get_linewidth_iterator(what_to_iterate,existing_styles=None,linewidths=None):

    if existing_styles is None:
        existing_styles = [ {} for _ in what_to_iterate]
    if linewidths is None:
        linewidths = [1.5,1.0,0.5]
    linewidths = bp.simple_cycler(linewidths)
    for i, _ in enumerate(what_to_iterate):
        existing_styles[i]['ls'] = linestyles[i]
    return existing_styles

def get_marker_iterator(what_to_iterate,existing_styles=None,markers=None):

    if existing_styles is None:
        existing_styles = [ {} for _ in what_to_iterate]
    if markers is None:
        markers = bp.markers.items()
    markers = bp.simple_cycler(markers)
    for i, _ in enumerate(what_to_iterate):
        existing_styles[i]['ls'] = markers[i]
    return existing_styles


def make_lighter(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    dl = 1-l
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l + dl * scale_l), s = s)
