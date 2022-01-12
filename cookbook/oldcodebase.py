import numpy as np
import bfmplot as bp
pl = bp.pl
markers = bp.markers
colors = list(bp.colors)
from copy import deepcopy

from de_omicron_wave.applied_modeling import _dts, _dte, t0, format_dates
import qsuite_config as cf

from datetime import date
from itertools import product

tomic = (date(2021,11,23) - date(*t0)).days

data = np.load('results.npy')

params = cf.external_parameters + cf.internal_parameters
param_names = list(map(lambda x: x[0], params))
param_values = list(map(lambda x: x[1], params))
param_indices = list(map(lambda x: list(range(len(x[1]))), params))

per_simulation_param_names = [
            'observable',
        ]

per_simulation_param_values = [ deepcopy(cf.observable_keys) ]
per_simulation_param_indices = [ list(range(len(vals))) for vals in per_simulation_param_values ]

param_names += per_simulation_param_names
param_values += per_simulation_param_values
param_indices += per_simulation_param_indices

#from rich import print
#print(param_names)
#print(param_values)
#print(param_indices)

def get_parameter_index(param_name):

    ndx = param_names.index(param_name)
    if ndx == -1:
        raise ValueError(f'Unknown Parameter "{param_name}"')

    return ndx

def get_parameter_values(param_name):
    return param_values[get_parameter_index(param_name)]

def get_parameter_value_index(param_name, value):

    pndx = get_parameter_index(param_name)
    ndx = param_values[pndx].index(value)
    if ndx == -1:
        raise ValueError(f'Unknown Parameter value "{value}"')

    return ndx

def get_curve_ndcs(parameters):
    print(parameters, param_names)
    if len(parameters) != len(param_names):
        raise ValueError(f'Not enough parameters provided to get curve')

    ndcs = []
    for name in param_names:
        if 'ind' in parameters[name]:
            ndcs.append(parameters[name]['ind'])
        else:
            ndx = get_parameter_value_index(name, parameters[name]['val'])
            ndcs.append(ndx)

    return tuple(ndcs)

def get_curve(parameters):
    return data[get_curve_ndcs(parameters)]



def observable_label(obs_name):
    labels = {
            ('incidence','C','total'): 'Tgl. Neue Fälle',
            ('incidence','H','total'): 'Tgl. Neuhospitalisierungen',
            ('prevalence','U','total'): 'Belegung ITS',
            ('prevalence','U','o'): 'Belegung ITS (Omikron)',
            ('variant_share','o'): 'Anteil Omikron',
            ('doubling_time','o'): 'Verdoppl.zeit (Tage)',
        }

    return labels[obs]

def update_min_max(minmax,vals):
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


def make_comparison_figure(what_to_iterate_on_columns,
                           what_to_iterate_on_rows,
                           what_to_iterate_on_axis,
                           sharex='none',
                           sharey='none',
                           what_to_keep_constant={},
                           colwidth=3,
                           rowheight=2.5,
                           x=None,
                           strip_axis=True,
                           styles_to_iterate_on_columns=[],
                           styles_to_iterate_on_rows=[],
                           styles_to_iterate_on_axis=[],
                           additional_plot_kwargs={},
                           ax = None,
                           nice_ticks=None,
                           xlim=(None,None),
                           ylim=(None,None),
                           automatically_adjust_xlim=True,
                           automatically_adjust_ylim=True,
                           format_dates=None,
                           get_row_label=None,
                           get_col_label=None,
                           ):


    nC = len(what_to_iterate_on_columns)
    nR = len(what_to_iterate_on_rows)

    if ax is None:
        wth = colwidth
        hght = rowheight
        fig, ax = pl.subplots(nR, nC, figsize=(nC*wth, nR*hght), sharex=sharex, sharey=sharey)

        if nC == 1 and nR == 1 :
            ax = np.array([ax])

        ax = ax.reshape(nR, nC)
    else:
        assert(ax.shape == (nR, nC))


    for irow, row in enumerate(what_to_iterate_on_rows):
        for icol, col in enumerate(what_to_iterate_on_columns):
            these_parameters = deepcopy(row)
            these_parameters.update(deepcopy(col))

            xminmax = [None,None]
            yminmax = [None,None]
            for icurve, curve in enumerate(what_to_iterate_on_axis):
                these_curve_parameters = deepcopy(these_parameters)
                these_curve_parameters.update(curve)
                these_curve_parameters.update(what_to_keep_constant)
                y = get_curve(these_curve_parameters)
                if x is None:
                    x = np.arange(len(y))

                xminmax = update_min_max(xminmax,x)
                yminmax = update_min_max(yminmax,y)


                plot_kwargs = deepcopy(additional_plot_kwargs)
                if len(styles_to_iterate_on_columns) > 0:
                    plot_kwargs.update(styles_to_iterate_on_columns[icol])
                if len(styles_to_iterate_on_rows) > 0:
                    plot_kwargs.update(styles_to_iterate_on_rows[irow])
                if len(styles_to_iterate_on_axis) > 0:
                    plot_kwargs.update(styles_to_iterate_on_axis[icurve])
                ax[irow, icol].plot(x,y,**plot_kwargs)

            this_xlim = list(xlim)
            this_ylim = list(ylim)

            for i in range(2):
                if this_xlim[i] is None:
                    this_xlim[i] = xminmax[i]
                if this_ylim[i] is None:
                    this_ylim[i] = yminmax[i]

            if automatically_adjust_xlim:
                ax[irow,icol].set_xlim(this_xlim)
            if automatically_adjust_ylim:
                ax[irow,icol].set_ylim(this_ylim)

            if nice_ticks is not None:
                bp.nice_ticks(ax[irow, icol],nice_ticks)

    for a in ax.flatten():
        if strip_axis:
            bp.strip_axis(a)
        if format_dates is not None:
            format_dates(a)

    return fig, ax

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
    linestyles = bp.simple_cycler(linesstyles)
    for i, _ in enumerate(what_to_iterate):
        existing_styles[i]['ls'] = linestyles[i]

def get_linewidth_iterator(what_to_iterate,existing_styles=None,linewidths=None):

    if existing_styles is None:
        existing_styles = [ {} for _ in what_to_iterate]
    if linewidths is None:
        linewidths = [1.5,1.0,0.5]
    linewidths = bp.simple_cycler(linewidths)
    for i, _ in enumerate(what_to_iterate):
        existing_styles[i]['ls'] = linestyles[i]

def get_marker_iterator(what_to_iterate,existing_styles=None,markers=None):

    if existing_styles is None:
        existing_styles = [ {} for _ in what_to_iterate]
    if linewidths is None:
        linewidths = [1.5,1.0,0.5]
    linewidths = bp.simple_cycler(linewidths)
    for i, _ in enumerate(what_to_iterate):
        existing_styles[i]['ls'] = linestyles[i]

def get_parameter_product(parameter_names,value_or_index):
    """Shortcut to get a subspace of the whole parameter product space"""
    assert(len(parameter_names) == len(value_or_index))

    these_param_indices = [ param_indices[get_parameter_index(name)] for name in parameter_names ]

    this_product = []
    for ndx in product(*these_param_indices):
        this_entry = {}
        for name, i, voi in zip(parameter_names, ndx, value_or_index):
            print(name, i, voi)
            if voi.startswith('i'):
                this_entry[name] = {'ind':i}
            elif voi.startswith('v'):
                this_entry[name] = {'val':get_parameter_values(name)[i]}

        this_product.append(this_entry)

    return this_product

def get_parameter_value_product(*parameter_names):
    """Shortcut to get a subspace of the whole parameter product space"""
    voi = ['v' for _ in parameter_names]
    return get_parameter_product(parameter_names, voi)

def get_parameter_index_product(*parameter_names):
    """Shortcut to get a subspace of the whole parameter product space"""
    voi = ['i' for _ in parameter_names]
    return get_parameter_product(parameter_names, voi)


if __name__ == "__main__":
    from rich import print

    what_to_iterate_on_columns = get_parameter_value_product('infectious_period', 'omicron_latent_period')
    what_to_iterate_on_rows = [
                { 'observable': {'val':('incidence','C','total') }, },
                { 'observable': {'val':('incidence','H','total') }, },
                { 'observable': {'val':('prevalence','U','total') }, },
            ]

    what_to_iterate_on_axis = [
                { 'omicron_base_probability_scaling': { 'ind': i} } for i in range(10)
            ]

    for iaxit in range(len(what_to_iterate_on_axis)):
        axit = what_to_iterate_on_axis[iaxit]
        axit['relative_additional_time_modulation_pairs'] = {'ind': 0}
        axit = deepcopy(axit)
        axit['relative_additional_time_modulation_pairs'] = {'ind': 1}
        what_to_iterate_on_axis.append(axit)

    what_to_keep_constant = {
                    'booster_scenario': {'val':'medium_rate_medium_reach_inf2FromDataInfBFromSymp_escape'},
                }

    dates = _dts(np.arange(129)+tomic)

    styles_to_iterate_on_columns = get_color_iterator(what_to_iterate_on_columns)
    styles_to_iterate_on_axis = [{'ls':['-','--'][int(np.round(i/len(what_to_iterate_on_axis)))]} for i in range(len(what_to_iterate_on_axis))]

    fig, ax = make_comparison_figure(
                           what_to_iterate_on_columns,
                           what_to_iterate_on_rows,
                           what_to_iterate_on_axis,
                           what_to_keep_constant=what_to_keep_constant,
                           x=dates,
                           format_dates=format_dates,
                           nice_ticks='y',
                           sharey='row',
                           styles_to_iterate_on_columns=styles_to_iterate_on_columns,
                           styles_to_iterate_on_axis=styles_to_iterate_on_axis,
                           )
    fig.tight_layout()

    pl.show()



