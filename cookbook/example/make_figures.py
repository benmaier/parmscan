import numpy as np
from generate_data import parameters, time
from parmscan import ScanPlotter, get_color_iterator
import parmscan as psc

import matplotlib as mpl


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
    data = np.load('data/result.npy')
    S = ScanPlotter(data,parameters)


    figs, axs, datas = S.make_comparison_figures(
                what_to_iterate_on_figures=S.prod(['frequency', 'amplitude']),
                what_to_iterate_on_rows=S.get('observable'),
                #what_to_iterate_on_columns=psc.add_markers(S.get('phase')),
                what_to_iterate_on_columns=S.get('phase'),
                what_to_iterate_on_axis=S.dprod(
                                                    psc.add_colors(S.get('y0')),
                                                    psc.add_linestyles(S.get('decay',[0,2])),
                                               ),
                x_parameter='time',
                get_fig_title=get_figure_title,
                get_fig_caption=get_figure_caption,
                get_y_label=get_y_label,
                get_col_label=get_col_label,
                construct_data_structure=True,
                #actually_plot_on_figures=False,
                sharey='row',
                sharex='all',
            )

    from bfmplot import pl

    #from rich import print
    #print(datas[0])

    import orjson
    with open('figures.json','wb') as f:
        f.write(orjson.dumps(datas,option=orjson.OPT_SERIALIZE_NUMPY))

    figs, axs, datas = S.make_comparison_figures(
                what_to_iterate_on_figures=S.prod(['frequency', 'amplitude']),
                what_to_iterate_on_rows=S.get('observable',[0]),
                what_to_iterate_on_columns=psc.add_markers(S.get('phase')),
                what_to_iterate_on_axis=S.dprod(
                                                    psc.add_colors(S.get('y0')),
                                                    psc.add_linestyles(S.get('decay',[0,2])),
                                               ),
                x_parameter='time',
                which_result_to_put_on_x_instead_of_parameter=S.get('observable',[1])[0],
                get_fig_title=get_figure_title,
                get_fig_caption=get_figure_caption,
                get_y_label=lambda pars: "position",
                x_label="momentum",
                get_col_label=get_col_label,
                construct_data_structure=True,
                actually_plot_on_figures=False,
            )
    pl.show()
