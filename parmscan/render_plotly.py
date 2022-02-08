import plotly.express
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import simplejson as json
from parmscan.tools import (
        hex_to_rgb_css,
    )

from rich import print

def _convert_share_ax(share):
    if share == 'none':
        return False
    elif share == 'col':
        return 'columns'
    elif share == 'row':
        return 'rows'
    elif share == True:
        return 'all'
    else:
        return share

def _map_marker(m):
    mp = {'s':'square',
          'd':'diamond',
          'o':'circle',
          'v':'triangle-down',
          '*':'star',
          '^':'triangle-up',
          '>':'triangle-right',
          'h':'hexagon',
          'p':'pentagon',
          'P':'cross',
          '<':'triangle-left',
          '8':'octagon',
          'H':'hexagon2',
          'X':'x',
          }
    return mp[m]

def _convert_plot_kwargs(kwargs):

    line_args = {}
    marker_args = {}
    mode = None
    if 'ls' in kwargs:
        if (kwargs['ls'].lower() == 'none' or kwargs['ls'] is None) and 'marker' in kwargs:
            mode = 'markers'
        elif 'marker' in kwargs:
            mode = 'lines+markers'
        else:
            mode = 'lines'
    elif 'marker' in kwargs:
        mode = 'lines+markers'
    else:
        mode = 'lines'

    if 'lines' in mode and 'ls' in kwargs:
        if kwargs['ls'] == '-.':
            line_args['dash'] = 'dashdot'
        elif kwargs['ls'] == ':':
            line_args['dash'] = 'dot'
        elif kwargs['ls'] == '--':
            line_args['dash'] = 'dash'

    if 'marker' in kwargs:
        marker_args['symbol'] = _map_marker(kwargs['marker'])

    if 'ms' in kwargs:
        marker_args['marker_size'] = kwargs['ms']

    if 'lw' in kwargs:
        line_args['width'] = kwargs['lw']

    if 'color' in kwargs:
        marker_args['color'] = kwargs['color']
        line_args['color'] = kwargs['color']

    if 'alpha' in kwargs:
        marker_args['opacity'] = kwargs['alpha']
        line_args['opacity'] = kwargs['alpha']

    return mode, line_args, marker_args


def _lbl(s):
    if isinstance(s,str):
        return s.replace('\n',', ')
    else:
        return s


def render_figure_as_plotly(figure,figure_layout=None):
    nC = figure['ncols']
    nR = figure['nrows']
    sharex = _convert_share_ax(figure['sharex'])
    sharey = _convert_share_ax(figure['sharey'])
    fig = make_subplots(
                        cols=nC,
                        rows=nR,
                        shared_yaxes=sharey,
                        shared_xaxes=sharex,
                        column_widths=nC*[figure['colwidth']],
                        row_heights=nR*[figure['rowheight']],
                        column_titles=list(map(_lbl,figure['column_titles'])),
                        #row_titles=list(map(_lbl,figure['y_labels'])),
                       )
    for iR, row in enumerate(figure['panels']):
        for iC, panel in enumerate(row):
            for ibound, bounds in enumerate(panel['bounds']):
                curve = panel['curves'][ibound]
                for bound in bounds:
                    mode, line, marker = _convert_plot_kwargs(bound['plot_args'])
                    fig.add_trace(
                                  go.Scatter(
                                        x=list(curve['x'])+\
                                          list(curve['x'])[::-1],
                                        y=list(bound['upper'])+\
                                          list(bound['lower'])[::-1],
                                        line={'color':'rgba(0,0,0,0)'},
                                        fill='toself',
                                        fillcolor=hex_to_rgb_css(line['color'],
                                                                 line['opacity'],
                                                                ),
                                        showlegend=False,
                                        hoverinfo='skip',
                                  ),
                                  row=iR+1,
                                  col=iC+1,
                        )

            showlegend = iR == 0 and iC == 0

            for curve in panel['curves']:
                mode, line, marker = _convert_plot_kwargs(curve['plot_args'])
                fig.add_trace(
                                go.Scatter(
                                    x=curve['x'],
                                    y=curve['y'],
                                    line=line,
                                    marker=marker,
                                    mode=mode,
                                    name=_lbl(curve['label']),
                                    showlegend=showlegend,
                              ),
                              row=iR+1,
                              col=iC+1,
                    )

    if figure_layout is None:
        figure_layout = {
                'template':'plotly_white',
                'font':{ 'family':'Helvetica,Arial,Free Sans,Tahoma,Nimbus Sans,Calibri',
                         'color': '#000',
                    },
            }
    fig.update_layout(
                title=_lbl(figure['title']),
                **figure_layout,
            )
    for col in range(1,nC+1):
        fig.update_xaxes(title_text=_lbl(figure['x_label']), row=nR, col=col)

    for row in range(1,nR+1):
        if row <= len(figure['y_labels']):
            fig.update_yaxes(title_text=_lbl(figure['y_labels'][row-1]), row=row, col=1)

    return fig

def render_figures_as_plotly(figures):
    return list(map(render_figure_as_plotly, figures))

if __name__=="__main__":
    with open('../cookbook/example/figures.json','r') as f:
        figures = json.load(f)


    fig = render_figure_as_plotly(figures[5])
    fig.show()

    with open('../cookbook/stochastic_example/figures.json','r') as f:
        figures = json.load(f)

    fig = render_figure_as_plotly(figures[0])
    fig.show()

