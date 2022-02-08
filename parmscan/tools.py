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
    styles = get_color_iterator(what_to_iterate,colors=colors)
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

def add_linestyles(what_to_iterate,linestyles=None):
    styles = get_linestyle_iterator(what_to_iterate,linestyles=linestyles)
    return add_style(what_to_iterate, styles)


def get_linewidth_iterator(what_to_iterate,existing_styles=None,linewidths=None):

    if existing_styles is None:
        existing_styles = [ {} for _ in what_to_iterate]
    if linewidths is None:
        linewidths = [1.5,1.0,0.5]
    linewidths = bp.simple_cycler(linewidths)
    for i, _ in enumerate(what_to_iterate):
        existing_styles[i]['ls'] = linewidths[i]
    return existing_styles

def add_linewidths(what_to_iterate,linewidths=None):
    styles = get_linewidth_iterator(what_to_iterate,linewidths=linewidths)
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

def hex_to_rgb(_hex):
    rgb = mpl.colors.ColorConverter.to_rgb(_hex)
    return rgb

def hex_to_rgb_css(_hex,alpha=1.):
    rgb = hex_to_rgb(_hex)
    r, g, b = rgb
    rgba = 'rgba(%d,%d,%d,%4.2f)' % (int(r*255), int(g*255), int(b*255), alpha)
    return rgba


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

def get_default_label(parmset):
    """Create a default label from a parameterset"""
    lbls = []
    for k, v in parmset.items():
        if isinstance(v['val'], (int, np.uint, float,np.float, np.int)):
            val = human_format(v['val'],precision=2)
        else:
            val = v['val']
        lbls.append("{} = {}".format(k,val))
    return "\n".join(lbls)

def human_format(num, precision=2):
    """Return numbers rounded to given precision and with sensuous suffixes.

    Parameters
    ==========
    num : float
        The number to humanify.
    precision : int, default : 2
        Number of decimal places.

    Return
    ======
    s : String
        Human readable string.
    """
    if np.isclose(num, 0):
        return "0"
    if abs(num) > 1:
        suffixes=['', 'k', 'M', 'G', 'T', 'P']
        m = sum([abs(num/1000.0**x) >= 1 for x in range(1, len(suffixes))])
        s = "%.{}f".format(precision) % (num/1000.0**m)
    elif abs(num) < 1:
        suffixes=['', 'm', 'μ', 'n', 'p', 'f']
        m = sum([abs(num*1000.0**x) <= 1 for x in range(1, len(suffixes))])
        s = "%.{}f".format(precision) % (num*1000.0**m)
    else:
        return "-1" if num == -1 else "1"

    while s[-1] == '0':
        s = s[:-1]
    if s[-1] == '.':
        s = s[:-1]

    return s + suffixes[m]



if __name__=="__main__":
    for num in [
                13e9,
                1.23234523652346346e4,
                4.5e-10,
                1e-3,
                1.,
                0.1,
                np.float(0.000040000000000001),
            ]:
        print(num, human_format(num))

