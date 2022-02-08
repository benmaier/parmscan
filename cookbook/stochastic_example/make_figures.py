import numpy as np
from generate_data import parameters, time
from parmscan import ScanPlotter, get_color_iterator
import parmscan as psc

import matplotlib as mpl

# get rid of measurement axis
parameters = parameters[1:]

# add the simulation axes
parameters.append(( 'observable', ['deflection','derivative']))
parameters.append(( 'time', time ))


def get_figure_title(pars):
    return "frequency = {0:4.2f}, amplitude = {1:4.2f}".format(pars['frequency']['val'],
                                                               pars['amplitude']['val'],
                                                              )
def get_figure_caption(pars):
    return ""

def get_y_label(pars):
    return pars['observable']['val']

def get_col_label(pars):
    return 'phase = {0:4.2f}'.format(pars['phase']['val'])


if __name__ == "__main__":
    data = np.load('data/result_percentiles.npz')
    res = data['50']
    S = ScanPlotter(res,parameters,
                [
                    [data['25'], data['75']],
                    [data['2.5'], data['97.5']],
                ],
            )

    figs, axs, datas = S.make_comparison_figures(
                what_to_iterate_on_figures=S.dprod(
                                                S.get('frequency',[1]),
                                                S.get('amplitude',[-1]),
                                            ),
                what_to_iterate_on_rows=S.get('observable'),
                #what_to_iterate_on_columns=psc.add_markers(S.get('phase')),
                what_to_iterate_on_columns=S.get('phase'),
                what_to_iterate_on_axis=\
                                            S.dprod(
                                                    psc.add_colors(S.get('y0',[0,3])),
                                                    psc.add_linestyles(S.get('decay',[0,2])),
                                               ),
                x_parameter='time',
                #what_to_keep_constant={'decay':{'ind':0}},
                nice_ticks='xy',
                #get_fig_title=get_figure_title,
                #get_fig_caption=get_figure_caption,
                #get_y_label=get_y_label,
                #get_col_label=get_col_label,
                construct_data_structure=True,
                #actually_plot_on_figures=False,
                sharey='row',
                sharex='all',
                plot_bounds='fill_between',
                #additional_fill_between_kwargs={'alpha':1}
                make_bound_color_lighter=False,
            )

    from bfmplot import pl
    import orjson


    with open('figures.json','wb') as f:
        f.write(orjson.dumps(datas,option=orjson.OPT_SERIALIZE_NUMPY))

    #import plotly.tools as tls
    #plotly_fig = tls.mpl_to_plotly(figs[0])
    #plotly_fig.show()

    pl.show()
