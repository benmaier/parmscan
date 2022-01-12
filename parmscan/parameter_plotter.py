import numpy as np
import bfmplot as bp
from bfmplot import pl
from copy import deepcopy

from itertools import product

from parmscan.tools import update_min_max


class ScanPlotter():

    def __init__(self,data,
                      parameter_pairs,
                      per_simulation_param_names=[],
                      per_simulation_param_values=[],
                ):

        self.params = parameter_pairs
        self.param_names = list(map(lambda x: x[0], params))
        self.param_values = list(map(lambda x: (x[1]), params))
        self.param_indices = list(map(lambda x: list(range(len(x[1]))), params))

        self.per_simulation_param_names = per_simulation_param_names
        self.per_simulation_param_values = per_simulation_param_values
        self.per_simulation_param_indices = [ list(range(len(vals))) for vals in per_simulation_param_values ]

        self.param_names += per_simulation_param_names
        self.param_values += per_simulation_param_values
        self.param_indices += per_simulation_param_indices


    def get_parameter_index(self,param_name):

        ndx = self.param_names.index(param_name)
        if ndx == -1:
            raise ValueError(f'Unknown Parameter "{param_name}"')

        return ndx

    def get_parameter_values(self,param_name):
        return self.param_values[self.get_parameter_index(param_name)]

    def get_parameter_value_index(self,param_name,value):

        pndx = self.set_parameter_index(param_name)
        ndx = self.param_values[pndx].index(value)
        if ndx == -1:
            raise ValueError(f'Unknown Parameter value "{value}"')

        return ndx

    def get_curve_ndcs(self,parameters,xname):
        if len(parameters)+1 != len(self.param_names):
            missing = set(self.param_names) - set(list(parameters.keys())+[xname])
            raise ValueError(f"Not enough parameters provided to get curve, you didn't supply {missing}")

        ndcs = []
        for name in self.param_names:
            if name == xname:
                ndcs.append(slice(None))
            else:
                if 'ind' in parameters[name]:
                    ndcs.append(parameters[name]['ind'])
                else:
                    ndx = self.get_parameter_value_index(name, parameters[name]['val'])
                    ndcs.append(ndx)

        return tuple(ndcs)

    def get_curve(self,parameters):
        return self.data[self.get_curve_ndcs(parameters)]



    def make_comparison_figures(self,
                                what_to_iterate_on_figure,
                                get_fig_label=None,
                                **kwargs,
                                ):
        """Wrapper for :func:`ScanPlotter.make_comparison_figure`"""
        figs, axs = [], []
        for fig_it in what_to_iterate_on_figure:
            fig, ax = make_comparison_figure(what_to_keep_constant=fig_it,**kwargs)
            if get_fig_label is not None:
                lbl = get_fig_label(fig_it)
                fig.suptitle(lbl)
            fig.tight_layout()
            figs.append(fig)
            axs.append(ax)

        return figs, axs


    def make_comparison_figure(
                               self,
                               what_to_iterate_on_columns,
                               what_to_iterate_on_rows,
                               what_to_iterate_on_axis,
                               x_parameter,
                               which_result_to_put_on_x_instead_of_parameter={},
                               sharex='none',
                               sharey='none',
                               what_to_keep_constant={},
                               colwidth=3,
                               rowheight=2.5,
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
                               format_x=None,
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

                    y = self.get_curve(these_curve_parameters,x_parameter)

                    if len(which_result_to_put_on_x_instead_of_parameter) > 0:
                        these_curve_parameters.update(which_result_to_put_on_x_instead_of_parameter)
                        x = self.get_curve(these_curve_parameters,x_parameter)
                    else:
                        x = self.get_parameter_values(x_parameter)

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
            if format_x is not None:
                format_x(a)

        return fig, ax

    def self.get_parameter_product(self,parameter_names,value_or_index):
        """Shortcut to get a subspace of the whole parameter product space"""
        assert(len(parameter_names) == len(value_or_index))

        these_param_indices = [ self.param_indices[self.get_parameter_index(name)] for name in parameter_names ]

        this_product = []
        for ndx in product(*these_param_indices):
            this_entry = {}
            for name, i, voi in zip(parameter_names, ndx, value_or_index):
                if voi.startswith('i'):
                    this_entry[name] = {'ind':i}
                elif voi.startswith('v'):
                    this_entry[name] = {'val':self.get_parameter_values(name)[i]}

            this_product.append(this_entry)

        return this_product

    def get_parameter_value_product(self,*parameter_names):
        """Shortcut to get a subspace of the whole parameter product space"""
        voi = ['v' for _ in parameter_names]
        return self.get_parameter_product(parameter_names, voi)

    def get_parameter_index_product(self,*parameter_names):
        """Shortcut to get a subspace of the whole parameter product space"""
        voi = ['i' for _ in parameter_names]
        return self.get_parameter_product(parameter_names, voi)


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



