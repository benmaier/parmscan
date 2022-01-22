import numpy as np
import bfmplot as bp
from bfmplot import pl
import matplotlib as mpl
from copy import deepcopy

from itertools import product

from parmscan.tools import (
        update_min_max,
        make_lighter,
        get_default_label,
        rgb_to_hex,
    )


import warnings

from parmscan.render_plotly import (
        render_figure_as_plotly,
        render_figures_as_plotly,
    )


class ScanPlotter():

    def __init__(self,data,
                      parameter_pairs,
                      pairs_of_bounds_from_smallest_to_largest=[],
                      warnings=True,
                ):

        self.data = data
        self.params = parameter_pairs
        self.param_names = list(map(lambda x: x[0], self.params))
        self.param_values = list(map(lambda x: list(x[1]), self.params))
        self.param_indices = list(map(lambda x: list(range(len(x[1]))), self.params))
        self.data_bounds = pairs_of_bounds_from_smallest_to_largest
        self.warnings = warnings


    def get_parameter_index(self,param_name):

        ndx = self.param_names.index(param_name)
        if ndx == -1:
            raise ValueError(f'Unknown Parameter "{param_name}"')

        return ndx

    def get_parameter_values(self,param_name):
        return self.param_values[self.get_parameter_index(param_name)]

    def get(self,param_name,indices=None):
        values = self.get_parameter_values(param_name)
        _indices = list(range(len(values)))
        if indices is None:
            indices = _indices
        these_params = [ { param_name: {'val':values[i], 'ind': _indices[i]}} for i in indices ]
        return these_params

    def get_reversed(self,param_name,indices=None):
        l = self.get(param_name, indices)
        return l[::-1]


    def get_value(self,parameterset,name):

        if 'val' in parameterset[name]:
            return parameterset[name]['val']

        ndx = parameterset[name]['ind']
        vals = self.get_parameter_values(name)
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
            too_many = set(list(parameters.keys())+[xname]) - set(self.param_names)
            if len(missing)>0:
                raise ValueError(f"Not enough parameters provided to get curve, you didn't supply {missing}")
            else:
                if self.warnings:
                    warnings.warn(f"You passed parameters that are not in your parameter list: {too_many}")

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

    def complete_parameter_set(self,
                               parameterset,
                               ):
        for k, v in parameterset.items():
            if 'val' not in v:
                v['val'] = self.get_value(parameterset,k)
            if 'ind' not in v:
                v['ind'] = self.get_parameter_value_index(k,v['val'])


    def make_comparison_figures_plotly(self,
                                    *args,
                                    **kwargs,
                                    ):
        kwargs['actually_plot_on_figures'] = False
        kwargs['construct_data_structure'] = True
        figs, axs, datas = self.make_comparison_figures(*args,**kwargs)
        return render_figures_as_plotly(datas)

    def make_comparison_figure_plotly(self,*args,**kwargs):
        kwargs['actually_plot_on_figures'] = False
        kwargs['construct_data_structure'] = True
        fig, ax, data = self.make_comparison_figure(self,*args,**kwargs)
        return render_figure_as_plotly(data)

    def make_comparison_figures(self,
                                what_to_iterate_on_figures,
                                get_fig_title=None,
                                get_fig_caption=None,
                                *args,
                                **kwargs,
                                ):
        """Wrapper for :func:`ScanPlotter.make_comparison_figure`"""

        if 'what_to_keep_constant' in kwargs:
            _what_to_keep_constant = kwargs.pop('what_to_keep_constant')
        else:
            _what_to_keep_constant = {}

        if 'actually_plot_on_figures' in kwargs:
            actually_plot_on_figures = kwargs['actually_plot_on_figures']
        else:
            actually_plot_on_figures = True

        figs, axs, datas = [], [], []

        for fig_it in what_to_iterate_on_figures:

            what_to_keep_constant = deepcopy(_what_to_keep_constant)

            what_to_keep_constant.update(fig_it)
            kwargs['what_to_keep_constant'] = what_to_keep_constant
            fig, ax, data = self.make_comparison_figure(*args,**kwargs)

            if get_fig_title is not None:
                lbl = get_fig_title(fig_it)
            else:
                lbl = get_default_label(fig_it)

            if actually_plot_on_figures:
                fig.suptitle(lbl)

            data['title'] = lbl

            if get_fig_caption is not None:
                cap = get_fig_caption(fig_it)
                data['caption'] = cap

            if actually_plot_on_figures:
                fig.tight_layout()
            figs.append(fig)
            axs.append(ax)
            datas.append(data)

        return figs, axs, datas


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
                               get_curve_label=None,
                               plot_bounds=False, # can be True, False, 'fill_between', or 'errorbar'; 'True' defaults to 'fill_between'
                               plot_bounds_only=False,
                               additional_fill_between_kwargs={'alpha':0.35},
                               make_bound_color_lighter=True,
                               additional_errorbar_kwargs={},
                               construct_data_structure=False,
                               actually_plot_on_figures=True,
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
        fill_between_kwargs.update(additional_fill_between_kwargs)

        errorbar_kwargs = {}
        errorbar_kwargs.update(additional_errorbar_kwargs)



        nC = len(what_to_iterate_on_columns)
        nR = len(what_to_iterate_on_rows)

        if ax is None:
            wth = colwidth
            hght = rowheight
            if actually_plot_on_figures:
                fig, ax = pl.subplots(nR, nC, figsize=(nC*wth, nR*hght), sharex=sharex, sharey=sharey)

                if nC == 1 and nR == 1 :
                    ax = np.array([ax])

                ax = ax.reshape(nR, nC)
            else:
                fig = None
                ax = None
        else:
            assert(ax.shape == (nR, nC))

        if x_label is None:
            x_label = x_parameter
        if x_label is None or x_label.upper() == 'NONE':
            x_label = None

        if sharex == True:
            sharex = 'all'
        if sharey == True:
            sharey = 'all'

        # construct references to limits
        # according to the rules of axis sharing
        shares = [sharex,sharey]
        all_minmax = [None,None]
        for ish, share in enumerate(shares):
            if share == 'all':
                # all entries in all_minmax refer to the same minmax
                all_minmax[ish] =  [ [[None,None]] * nC ] * nR
            elif share == 'none':
                # all entries in all_minmax refer to a unique minmax
                all_minmax[ish] = [ [[None,None] for _ in range(nC)]\
                                    for __ in range(nR) ]
            elif share == 'col':
                # entries in all_minmax refer to a unique minmax for each column
                all_minmax[ish] = [ [[None,None] for _ in range(nC)] ] * nR
            elif share == 'row':
                # entries in all_minmax refer to a unique minmax for each column
                all_minmax[ish] = [ [[None,None]] * nC for _ in range(nR) ]

        figure = {
                    'ncols': nC,
                    'nrows': nR,
                    'colwidth': colwidth,
                    'sharex':sharex,
                    'sharey':sharey,
                    'rowheight': rowheight,
                    'panels': [ [ None for col in range(nC) ] for row in range(nR)  ],
                    'column_titles': [],
                    'x_label': x_label,
                    'y_labels': [],
                 }

        for irow, row in enumerate(what_to_iterate_on_rows):
            for icol, col in enumerate(what_to_iterate_on_columns):
                these_parameters = deepcopy(row)
                these_parameters.update(deepcopy(col))

                xminmax = all_minmax[0][irow][icol]
                yminmax = all_minmax[1][irow][icol]

                if construct_data_structure:
                    curves = []
                    bounds = []

                for icurve, curve in enumerate(what_to_iterate_on_axis):

                    these_curve_parameters = deepcopy(these_parameters)
                    these_curve_parameters.update(curve)
                    these_curve_parameters.update(what_to_keep_constant)
                    self.complete_parameter_set(these_curve_parameters)

                    y = self.get_curve(these_curve_parameters,x_parameter)

                    if len(which_result_to_put_on_x_instead_of_parameter) > 0:
                        tmp = deepcopy(these_curve_parameters)
                        tmp.update(which_result_to_put_on_x_instead_of_parameter)
                        x = self.get_curve(tmp,x_parameter)
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

                    for parameterset in [row, col, curve]:
                        this_set = deepcopy(parameterset)
                        for this_voi in this_set.values():
                            for key in ['val','v','i','ind']:
                                if key in this_voi:
                                    this_voi.pop(key)

                            plot_kwargs.update(this_voi)

                    if get_curve_label:
                        curve_label = get_curve_label(these_curve_parameters)
                    else:
                        curve_label = get_default_label(curve)

                    if not plot_bounds_only:
                        if actually_plot_on_figures:
                            plot_handle, = ax[irow, icol].plot(x,y,label=curve_label,**plot_kwargs)
                        if construct_data_structure:
                            if 'color' not in plot_kwargs:
                                prop_cycle = mpl.rcParams['axes.prop_cycle']
                                colors = prop_cycle.by_key()['color']
                                color = colors[icurve]
                                plot_kwargs['color'] = color
                            this_curve = {'x':x,
                                          'y':y,
                                          'plot_args':plot_kwargs,
                                          'label':curve_label,
                                          'curve_parameters':these_curve_parameters,
                                          }
                            curves.append(this_curve)
                    else:
                        plot_handle = None

                    if plot_bounds:

                        if actually_plot_on_figures and plot_handle is not None:
                            color = plot_handle.get_color()
                        else:
                            if 'color' in plot_kwargs:
                                color = plot_kwargs['color']
                            else:
                                prop_cycle = mpl.rcParams['axes.prop_cycle']
                                colors = prop_cycle.by_key()['color']
                                color = colors[icurve]

                        Nbounds = len(self.data_bounds)
                        rgb = mpl.colors.ColorConverter.to_rgb(color)
                        these_bounds = []
                        for i in range(Nbounds-1,-1,-1):
                            if make_bound_color_lighter:
                                betw_color = make_lighter(rgb,((i+1+1)/(Nbounds+2)))
                                errbar_color = make_lighter(rgb,((i+1)/(Nbounds+2)))
                                fill_between_kwargs.update({'alpha':1})
                            else:
                                betw_color = rgb_to_hex(rgb)
                                errbar_color = rgb_to_hex(rgb)

                            upper = self.get_upper_bound(i,these_curve_parameters,x_parameter)
                            lower = self.get_lower_bound(i,these_curve_parameters,x_parameter)
                            yminmax = update_min_max(yminmax,upper)
                            yminmax = update_min_max(yminmax,lower)

                            if plot_bounds == 'fill_between':
                                if actually_plot_on_figures:
                                    ax[irow, icol].fill_between(x,lower,upper,
                                                                color=betw_color,
                                                                **fill_between_kwargs,
                                                               )
                                if construct_data_structure:
                                    _plot_kwargs = deepcopy(fill_between_kwargs)
                                    _plot_kwargs['color'] = betw_color
                                    these_bounds.append({
                                            'lower': lower,
                                            'upper': upper,
                                            'kind': 'fill_between',
                                            'plot_args': _plot_kwargs,
                                        })
                            elif plot_bounds == 'errorbar':
                                if actually_plot_on_figures:
                                    ax[irow, icol].errorbar(x,y,
                                                            yerr=np.array([y-lower,upper-y]),
                                                            color=errbar_color,
                                                            **errorbar_kwargs,
                                                            )
                                if construct_data_structure:
                                    _plot_kwargs = deepcopy(fill_between_kwargs)
                                    _plot_kwargs['color'] = errbar_color
                                    these_bounds.append({
                                            'lower': lower,
                                            'upper': upper,
                                            'kind': 'errorbars',
                                            'color': errbar_color,
                                            'plot_args': _plot_kwargs,
                                        })

                        bounds.append(these_bounds)
                if construct_data_structure:
                    figure['panels'][irow][icol] = {'curves':curves,'bounds':bounds}


                this_xlim = list(xlim)
                this_ylim = list(ylim)

                for i in range(2):
                    if this_xlim[i] is None:
                        this_xlim[i] = xminmax[i]
                    if this_ylim[i] is None:
                        this_ylim[i] = yminmax[i]

                if actually_plot_on_figures:
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
                        if actually_plot_on_figures:
                            ax[irow,icol].set_xlabel(x_label)

                if irow == 0:
                    if get_col_label is not None:
                        col_label = get_col_label(col)
                    else:
                        col_label = get_default_label(col)
                    if actually_plot_on_figures:
                        ax[0,icol].set_title(col_label,fontsize='medium',loc='right')
                    figure['column_titles'].append(col_label)

            if get_y_label is not None:
                ylabel = get_y_label(row)
            else:
                ylabel = get_default_label(row)
            if actually_plot_on_figures:
                ax[irow,0].set_ylabel(ylabel)
            if construct_data_structure:
                figure['y_labels'].append(ylabel)

        if actually_plot_on_figures:
            for a in ax.flatten():
                if strip_axis:
                    bp.strip_axis(a)
                if format_x is not None:
                    format_x(a)

        if construct_data_structure:
            for irow, row in enumerate(what_to_iterate_on_rows):
                for icol, col in enumerate(what_to_iterate_on_columns):
                    figure['panels'][irow][icol]['data_bounds'] = {
                                'x': all_minmax[0][irow][icol],
                                'y': all_minmax[1][irow][icol],
                            }

        return fig, ax, figure

    def parameter_product(self,parameter_names,reverse=False):
        """Shortcut to get a subspace of the whole parameter product space"""

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
            for name, i in zip(parameter_names, ndx):
                    this_entry[name] = {
                        'ind':i,
                        'val':self.get_parameter_values(name)[i],
                    }

            this_product.append(this_entry)

        return this_product

    def get_parameter_value_product(self,*parameter_names,reverse=False):
        """Shortcut to get a subspace of the whole parameter product space"""
        voi = ['val' for _ in parameter_names]
        return self.get_parameter_product(parameter_names, voi, reverse)

    def get_parameter_index_product(self,*parameter_names,reverse=False):
        """Shortcut to get a subspace of the whole parameter product space"""
        voi = ['ind' for _ in parameter_names]
        return self.get_parameter_product(parameter_names, voi, reverse)

    def parameter_dict_product(self,*parameter_dicts,reverse=False):
        """Construct a product of multiple lists of parameters"""

        if isinstance(reverse,bool):
            reverse = [ reverse for _ in range(len(parameter_dicts)) ]

        new_dict_list = []
        for dlist, rev in zip(parameter_dicts, reverse):
            if rev:
                new_dict_list.append(dlist[::-1])
            else:
                new_dict_list.append(dlist)

        new_list = []
        for dicts in product(*new_dict_list):
            new_dict = deepcopy(dicts[0])
            for entry in dicts[1:]:
                new_dict.update(entry)
            new_list.append(new_dict)

        return new_list

    def prod(self,*args,**kwargs):
        """Short for ``self.parameter_product``"""
        return self.parameter_product(*args,**kwargs)

    def dprod(self,*args,**kwargs):
        """Short for ``self.parameter_dict_product``"""
        return self.parameter_dict_product(*args,**kwargs)
