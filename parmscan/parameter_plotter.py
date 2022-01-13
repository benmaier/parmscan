import numpy as np
import bfmplot as bp
from bfmplot import pl
import matplotlib as mpl
from copy import deepcopy

from itertools import product

from parmscan.tools import (
        update_min_max,
        make_lighter,
    )


class ScanPlotter():

    def __init__(self,data,
                      parameter_pairs,
                      pairs_of_data_representing_error_bounds_from_highest_to_lowest=[],
                ):

        self.data = data
        self.params = parameter_pairs
        self.param_names = list(map(lambda x: x[0], self.params))
        self.param_values = list(map(lambda x: (x[1]), self.params))
        self.param_indices = list(map(lambda x: list(range(len(x[1]))), self.params))
        self.data_bounds = pairs_of_data_representing_error_bounds_from_highest_to_lowest


    def get_parameter_index(self,param_name):

        ndx = self.param_names.index(param_name)
        if ndx == -1:
            raise ValueError(f'Unknown Parameter "{param_name}"')

        return ndx

    def get_parameter_values(self,param_name):
        return self.param_values[self.get_parameter_index(param_name)]

    def get_value(self,parameterset,name):

        if 'val' in parameterset[name]:
            return parameterset[name]['val']

        ndx = parameterset[name]['ind']
        vals = get_parameter_values(name)
        val = vals[ndx]

        return val

    def get_parameter_value_index(self,param_name,value):

        pndx = self.get_parameter_index(param_name)
        ndx = self.param_values[pndx].index(value)
        if ndx == -1:
            raise ValueError(f'Unknown Parameter value "{value}"')

        return ndx

    def get_curve_ndcs(self,parameters,xname):
        if (len(parameters)+1) != len(self.param_names):
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

    def get_curve(self,parameters,xname):
        return self.data[self.get_curve_ndcs(parameters,xname)]

    def get_upper_bound(self,boundindex,parameters,xname):
        data = self.data_bounds[boundindex][1]
        return data[self.get_curve_ndcs(parameters,xname)]

    def get_lower_bound(self,boundindex,parameters,xname):
        data = self.data_bounds[boundindex][0]
        return data[self.get_curve_ndcs(parameters,xname)]


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
                               what_to_keep_constant={},
                               which_result_to_put_on_x_instead_of_parameter={},
                               sharex='none',
                               sharey='none',
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
                               get_y_label=None,
                               x_label=None,
                               get_col_label=None,
                               plot_bounds=False, # can be True, False, 'fill_between', or 'errorbar'; 'True' defaults to 'fill_between'
                               plot_bounds_only=False,
                               additional_fill_between_kwargs={},
                               additional_errorbar_kwargs={},
                              ):
        """
        Create a grid figure where certain parameter combinations
        are iterated over columns, others are iterated over
        rows, and then others are iterated over as curves
        on the axis.

        Parameters
        ==========
        what_to_iterate_on_columns : list of dict
            A list of parameter combinations.
            Every column of the figure is mapped
            to the corresponding entry in this list.
            Results displayed in this column have
            been generated with the parameter values
            given in this entry.
        what_to_iterate_on_rows : list of dict
            A list of parameter combinations.
            Every row of the figure is mapped
            to the corresponding entry in this list.
            Results displayed in this row have
            been generated with the parameter values
            given in this entry.
        what_to_iterate_on_axis : list of dict
            A list of parameter combinations.
            Every curve of this axis is mapped
            to the corresponding entry in this list.
        x_parameter : hashable type
            The parameter which to put on the x-axis.
        what_to_keep_constant : dict:
            A parameter dictionary that contains the
            parameter combination of the remaining,
            as of yet un-iterated parameters.
            These parameter values will be constant over
            the whole figure.
        which_result_to_put_on_x_instead_of_parameter : dict, default = {}
            A parameter dictionary that contains the
            parameter combination of the result that's
            supposed to go on the x-axis instead of
            ``x_parameter``. This doesn't have to
            be a full parameter combination, it can
            be a subset--The respective curve
            parameter combination will be updated
            with this value.
        sharex : str, default = 'none'
            will be passed to plt.subplots.
        sharey : str, default = 'none'
            will be passed to plt.subplots.

        strip_axis : bool, default = True
            Whether or not to strip the axes
            (remove the rectangular frame)


        Returns
        =======
        fig : matplotlib.Figure
            The resulting figure
        ax : numpy.ndarray of matplotlib.Axes
            The resulting numpy array containing the axes.
        """

        fill_between_kwargs = {
                    'edgecolor':'None',
                }

        errorbar_kwargs = {}
        errorbar_kwargs.update(additional_errorbar_kwargs)


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
                        print(icurve, styles_to_iterate_on_axis)
                        plot_kwargs.update(styles_to_iterate_on_axis[icurve])

                    for parametersets in [row, col, curve]:
                        this_set = deepcopy(parameterset)
                        for this_voi in this_set.values():
                            for key in ['val','v','i','ind']:
                                if key in this_voi:
                                    this_voi.pop(key)
                            plot_kwargs.update(this_voi)

                    if not plot_bounds_only:
                        plot_handle, = ax[irow, icol].plot(x,y,**plot_kwargs)
                    else:
                        plot_handle = None

                    if plot_bounds:

                        if plot_handle is not None:
                            color = plot_handle.get_color()
                        else:
                            if 'color' in plot_kwargs:
                                color = plot_kwargs
                            else:
                                prop_cycle = mpl.rcParams['axes.prop_cycle']
                                colors = prop_cycle.by_key()['color']
                                color = colors[icurve]

                        Nbounds = len(self.data_bounds)
                        rgb = mpl.colors.ColorConverter.to_rgb(color)
                        for i in range(Nbounds-1,-1,-1):
                            betw_color = make_lighter(rgb,((i+1+1)/(Nbounds+2)))
                            errbar_color = make_lighter(rgb,((i+1)/(Nbounds+2)))

                            upper = self.get_upper_bound(i,these_curve_parameters,x_parameter)
                            lower = self.get_lower_bound(i,these_curve_parameters,x_parameter)

                            if plot_bounds == 'fill_between':
                                ax[irow, icol].fill_between(x,lower,upper,color=betw_color,**fill_between_kwargs)
                            elif plot_bounds == 'errorbar':
                                ax[irow, icol].errorbar(x,y,yerr=np.array([y-lower,upper-y]),color=errbar_color,**errorbar_kwargs)

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

                if irow == nR-1:
                    if x_label is None:
                        x_label = x_parameter
                    if x_label.upper() != 'NONE':
                        ax[irow,icol].set_xlabel(x_label)

                if irow == 0:
                    if get_col_label is not None:
                        col_label = get_col_label(col)
                        ax[0,icol].set_title(col_label,fontsize='medium',loc='right')

            ylabel = get_y_label(row)
            ax[irow,0].set_ylabel(ylabel)

        for a in ax.flatten():
            if strip_axis:
                bp.strip_axis(a)
            if format_x is not None:
                format_x(a)

        return fig, ax

    def get_parameter_product(self,parameter_names,value_or_index,reverse=False):
        """Shortcut to get a subspace of the whole parameter product space"""

        assert(len(parameter_names) == len(value_or_index))
        if isinstance(reverse,bool):
            reverse = [ reverse for _ in range(len(parameter_names)) ]

        these_param_indices = []
        for name, rev in zip(parameter_names, reverse):
            ndcs = self.param_indices[self.get_parameter_index(name)]
            if rev:
                ndcs = ndcs[::-1]
            these_param_indices.append(ndcs)

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

    def get_parameter_value_product(self,*parameter_names,reverse=False):
        """Shortcut to get a subspace of the whole parameter product space"""
        voi = ['v' for _ in parameter_names]
        return self.get_parameter_product(parameter_names, voi, reverse)

    def get_parameter_index_product(self,*parameter_names,reverse=False):
        """Shortcut to get a subspace of the whole parameter product space"""
        voi = ['i' for _ in parameter_names]
        return self.get_parameter_product(parameter_names, voi, reverse)


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



