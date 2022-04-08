#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MWJ van Es, 2022
inspired by M. Fabus (https://gitlab.com/marcoFabus/fabus2022_harmonics/-/blob/main/app.py)
"""

import matplotlib.pyplot as plt
import dash
from dash.dependencies import Input, Output, State
import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc


def card_amp(header, name, props):
    card_content = [
        dbc.CardBody(
            [
                html.H4(header, className="card-title", id=header),
                html.Div(
                    style={'width': '90%'},
                    children=[

                        dbc.Row(
                            [

                                dbc.Col(
                                    html.Div(
                                        children=[html.P('Channel 1, freq 1'),
                                                  html.P('Channel 1, freq 2'),
                                                  html.P('Channel 2, freq 1'),
                                                  html.P('Channel 2, freq 2')]
                                    ),
                                    width=4
                                ),

                                dbc.Col(
                                    [
                                        html.Div(
                                            style={'padding-right': '0%', 'width': '120%'},
                                            children=[dcc.Slider(
                                                id=name[0],
                                                min=props[0],
                                                max=props[1],
                                                value=props[2][0],
                                                step=props[3],
                                                marks=props[4],
                                            ),
                                                dcc.Slider(
                                                    id=name[1],
                                                    min=props[0],
                                                    max=props[1],
                                                    value=props[2][1],
                                                    step=props[3],
                                                    marks=props[4]
                                                ),
                                                dcc.Slider(
                                                    id=name[2],
                                                    min=props[0],
                                                    max=props[1],
                                                    value=props[2][2],
                                                    step=props[3],
                                                    marks=props[4],
                                                ),
                                                dcc.Slider(
                                                    id=name[3],
                                                    min=props[0],
                                                    max=props[1],
                                                    value=props[2][3],
                                                    step=props[3],
                                                    marks=props[4],
                                                ), ])

                                    ],
                                    width=8)
                            ])
                    ]
                )
            ]
        )
    ]

    return card_content

def card_freq(header, name, props):
    card_content = [
        dbc.CardBody(
            [
                html.H4(header, className="card-title", id=header),
                html.Div(
                    style={'width': '90%'},
                    children=[

                        dbc.Row(
                            [

                                dbc.Col(
                                    html.Div(
                                        children=[html.P('Frequency component 1'),
                                                  html.P('Frequency component 2')]
                                    ),
                                    width=4
                                ),

                                dbc.Col(
                                    [
                                        html.Div(
                                            style={'padding-right': '0%', 'width': '120%'},
                                            children=[dcc.Slider(
                                                id=name[0],
                                                min=props[0],
                                                max=props[1],
                                                value=props[2][0],
                                                step=props[3],
                                                marks=props[4],
                                            ),
                                                dcc.Slider(
                                                    id=name[1],
                                                    min=props[0],
                                                    max=props[1],
                                                    value=props[2][1],
                                                    step=props[3],
                                                    marks=props[4],
                                                ), ])

                                    ],
                                    width=8)
                            ])
                    ]
                )
            ]
        )
    ]

    return card_content


radioitems = html.Div(
    [
        dbc.Label("Examples"),
        dbc.RadioItems(
            options=[
                {"label": "Example 1", "value":1},
                {"label": "Example 2", "value":2},
            ],
            value=1,
            id="example_button",
        ),
    ]
)

# %% Example plotly figure
headers = ['Frequency (Hz)', 'Condition 1', 'Condition 2']
names = [str(i) for i in range(10)]

marks_amp = {str(x): {'label': str(round(x, 2)), 'style': {'color': 'black'}} for x in np.linspace(0, 2, 9)}
marks_f = {str(int(x)): {'label': str(round(x)), 'style': {'color': 'black'}} for x in np.linspace(0, 20, 5)}

props_amp = [0, 2, [1.5, 0, 0.75, 0], 0.25, marks_amp]
props_f = [0, 5, [10, 0], 0.5, marks_f]

instructions = html.Div(
    [
        dbc.Button("Instructions", id="open-fs", size="lg"),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Instructions")),
                dbc.ModalBody(
                    html.Div([
                        html.A("Accompanying publication",
                               href='https://doi.org/10.1101/2022.02.07.479399', target="_blank"),
                        html.H5(""),
                        html.P(
                            "Paragraph\
                            One "),
                        html.H5(" "),
                        html.P("\
                    Paragraph \
                    Two"),
                    ])),
            ],
            id="modal-fs",
            fullscreen=False,
            scrollable=True,
            size="lg",
        ),
    ]
)

app = dash.Dash(__name__, title='RepresentationalDynamics', external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        html.H1('Representational Dynamics Simulator'),

        dbc.Row(
            [
                dbc.Col(dbc.Card(card_freq(headers[0], names[:2], props_f), style={"width": "20vw"}, color="light",
                                 inverse=False), width=3),
                dbc.Col(dbc.Card(card_amp(headers[1], names[2:6], props_amp), style={"width": "20vw"}, color="light",
                                 inverse=False), width=3),
                dbc.Col(dbc.Card(card_amp(headers[2], names[6:10], props_amp), style={"width": "20vw"}, color="light",
                                 inverse=False), width=3),
                dbc.Col(children=[instructions, radioitems])
            ],
        ),
        dbc.Row([dcc.Graph(id='graph', style={'width': '60vw', 'height': '70vh'})])
    ]
)


@app.callback(
    Output("modal-fs", "is_open"),
    Input("open-fs", "n_clicks"),
    State("modal-fs", "is_open"),
)
def toggle_modal(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    [Output('graph', 'figure'),
    [Input(x, 'value') for x in names], )
def update_figure(f1=10, f2=0, s1ch1f1=1.5, s1ch1f2=0, s1ch2f1=0.75, s1ch2f2=0, s2ch1f1=0, s2ch1f2=0, s2ch2f1=0,
                  s2ch2f2=0, example=1):
    # define standard parameters
    t = np.arange(0.001, 0.101, 0.001)
    if example == 1:
        # frequency
        f1 = 10.0
        f2 = 0

        # amplitude
        s1ch1f1 = 1.5  # magnitude of signal 1, channel 1, frequency component 1
        s1ch1f2 = 0  # magnitude of signal 1, channel 1, frequency component 2
        s1ch2f1 = 0.5 * s1ch1f1  # magnitude of signal 1, channel 2, frequency component 1
        s1ch2f2 = 0  # magnitude of signal 1, channel 2, frequency component 2
        s2ch1f1 = 0  # magnitude of signal 2, channel 1, frequency component 1
        s2ch1f2 = 0  # magnitude of signal 2, channel 1, frequency component 2
        s2ch2f1 = 0  # magnitude of signal 2, channel 2, frequency component 1
        s2ch2f2 = 0  # magnitude of signal 2, channel 2, frequency component 2

        ticks_amp = np.linspace(-2, 2, 5)
        ticks_pow = np.linspace(0, 2, 5)
        ticks_mi = np.linspace(0, 6, 4)
        ticks_mipow = np.linspace(0, 3, 4)

        ylim1 = (0, 6)
        ylim2 = (0, 3)
    elif example == 2:
        # This example has multiple frequncy components:
        f1 = 10.0
        f2 = 1.5 * f1

        s1ch1f1 = 0.5  # magnitude of signal 1, channel 1, frequency component 1
        s1ch1f2 = 0.77  # magnitude of signal 1, channel 1, frequency component 2
        s1ch2f1 = 0.5 * s1ch1f1  # magnitude of signal 1, channel 2, frequency component 1
        s1ch2f2 = 1.5  # magnitude of signal 1, channel 2, frequency component 2
        s2ch1f1 = 0.09  # magnitude of signal 2, channel 1, frequency component 1
        s2ch1f2 = 0.9 * 0.6  # magnitude of signal 2, channel 1, frequency component 1
        s2ch2f1 = s2ch1f1  # magnitude of signal 2, channel 1, frequency component 1
        s2ch2f2 = s2ch1f2  # magnitude of signal 2, channel 1, frequency component 1

        ticks_amp = np.linspace(-2, 2, 5)
        ticks_pow = np.linspace(0, 2, 5)
        ticks_mi = np.linspace(0, 1.5, 4)
        ticks_mipow = np.linspace(0, 0.5, 6)

        ylim1 = (0, 1.5)
        ylim2 = (0, 0.5)

    theta = np.array([[-np.pi / 2, -np.pi / 2], [-np.pi / 2, -np.pi / 2]])
    theta0 = theta
    s = np.array([0.5, 0.5])  # noise for channel 1 and channel 2
    Sig = np.diag(s) ** 2
    a = np.array(((s1ch1f1, s1ch1f2), (s1ch2f1, s1ch2f2)))
    a0 = np.array(((s2ch1f1, s2ch1f2), (s2ch2f1, s2ch2f2)))
    f = np.array([f1, f2])

    # signals
    xa = []
    x0 = []
    for k in range(2):
        xa.append(np.matmul(np.expand_dims(a[k, :], 0), np.cos(
            2 * np.pi * np.expand_dims(f, 1) * np.transpose(np.expand_dims(t, 1)) + np.expand_dims(theta[k, :], 1)))[
                      0])  # first channel, first condition
        x0.append(np.matmul(np.expand_dims(a0[k, :], 0), np.cos(
            2 * np.pi * np.expand_dims(f, 1) * np.transpose(np.expand_dims(t, 1)) + np.expand_dims(theta0[k, :], 1)))[
                      0])

    # now we need to map these two channel signals back to the format given in the paper:
    A_omega = []
    mu_omega = []
    phi_omega = []
    phi_mean = []
    for ifreq in np.arange(2):
        A_omega.append(0.5 * np.diag(np.sqrt(a[:, ifreq] ** 2 + a0[:, ifreq] ** 2 +
                                             2 * a[:, ifreq] * a0[:, ifreq] * np.cos(
            theta[:, ifreq] - theta0[:, ifreq] + np.array((-np.pi, -np.pi))))))
        mu_omega.append(0.5 * np.diag(np.sqrt(a[:, ifreq] ** 2 + a0[:, ifreq] ** 2 +
                                              2 * a[:, ifreq] * a0[:, ifreq] * np.cos(
            theta[:, ifreq] - theta0[:, ifreq]))))
        phi_omega.append(
            np.arctan2(a[:, ifreq] * np.sin(theta[:, ifreq]) + a0[:, ifreq] * np.sin(theta0[:, ifreq] + np.pi),
                       a[:, ifreq] * np.cos(theta[:, ifreq]) + a0[:, ifreq] * np.cos(
                           theta0[:, ifreq] + np.pi)))
        phi_mean.append(-np.pi / 2 + np.arctan2(
            a[:, ifreq] * np.sin(theta[:, ifreq] + np.pi / 2) + a0[:, ifreq] * np.sin(theta0[:, ifreq] + np.pi / 2),
            a[:, ifreq] * np.cos(theta[:, ifreq] + np.pi / 2) + a0[:, ifreq] * np.cos(theta0[:, ifreq] + np.pi / 2)))

    # find the information content terms:
    c_b = 0
    if len(f[f > 0]) == 1:
        Sigma = Sig
    else:
        Sigma = Sig + Sig  # broadband noise: sum over both frequency bands
    r_b = []
    psi = []
    for ifreq in np.arange(2):
        c_b = c_b + np.trace(np.matmul(np.matmul(A_omega[ifreq], np.linalg.inv(Sigma)), A_omega[ifreq]) * np.cos(
            np.expand_dims(phi_omega[ifreq], 1) - phi_omega[ifreq]))  # equal to above expression
        temp1 = np.trace(np.matmul(np.matmul(A_omega[ifreq], np.linalg.inv(Sigma)),
                                   A_omega[ifreq] * np.cos(np.expand_dims(phi_omega[ifreq], 1) + phi_omega[ifreq])))
        temp2 = np.trace(np.matmul(np.matmul(A_omega[ifreq], np.linalg.inv(Sigma)),
                                   A_omega[ifreq] * np.sin(np.expand_dims(phi_omega[ifreq], 1) + phi_omega[ifreq])))
        r_b.append(np.sqrt(temp1 ** 2 + temp2 ** 2))
        psi.append(np.arctan2(temp2, temp1))

    # and cross-frequency components:
    posmin = [1, -1]
    for i in np.arange(2):
        temp1 = np.trace(np.matmul(np.matmul(A_omega[0], np.linalg.inv(Sigma)),
                                   A_omega[1] * np.cos(np.expand_dims(phi_omega[0], 1) + posmin[i] * phi_omega[1])))
        temp2 = np.trace(np.matmul(np.matmul(A_omega[0], np.linalg.inv(Sigma)),
                                   A_omega[1] * np.sin(np.expand_dims(phi_omega[0], 1) + posmin[i] * phi_omega[1])))
        r_b.append(2 * np.sqrt(temp1 ** 2 + temp2 ** 2))
        psi.append(np.arctan2(temp2, temp1))
    r_b = np.array(r_b)

    infotermest = c_b
    freqs_all = np.concatenate((2 * f, np.array([np.sum(f)]), np.diff(f)))
    for ifreq in np.arange(4):
        infotermest = infotermest + r_b[ifreq] * np.cos(2 * np.pi * freqs_all[ifreq] * t + psi[ifreq])

    # Figure parameters
    ylim0 = (0, 2)
    ticks_time = np.linspace(0, 0.1, 3)
    ticks_freq = np.linspace(0, 40, 3)

    if not np.logical_or(example == 1, example == 2):
        ticks_amp = np.linspace(np.floor(np.min((np.min(xa[0]) - s[0], np.min(xa[1]) - s[1]))),
                                np.ceil(np.max((np.max(xa[0]) + s[0], np.max(xa[1]) + s[1]))),
                                np.diff((np.floor(np.min((np.min(xa[0]) - s[0], np.min(xa[1]) - s[1]))),
                                         np.ceil(np.max((np.max(xa[0]) + s[0], np.max(xa[1]) + s[1])))))[0] + 1)
        ticks_pow = np.linspace(0, np.max((np.max(a), np.max(a0))), 2 * np.max((np.max(a), np.max(a0))) + 1)
        ticks_mi = np.linspace(0, np.ceil(np.max(infotermest)), 2 * np.ceil(np.max(infotermest)) + 1)
        ticks_mipow = np.linspace(0, np.ceil(np.max(r_b)), 2 * np.ceil(np.max(r_b)) + 1)
        ylim1 = (0, np.ceil(np.max(infotermest)))
        ylim2 = (0, np.ceil(np.max(r_b)))

    # %% Plot everything
    fig = plt.figure()
    for k in range(3):
        ax1 = fig.add_subplot(3, 2, 2 * (k + 1) - 1)
        if k < 2:
            plt.plot(t, x0[k], 'k', linewidth=2)
            plt.fill_between(t, x0[k] - s[k], x0[k] + s[k], facecolor='gray', alpha=0.3)
            plt.plot(t, xa[k], 'b', linewidth=2)
            plt.fill_between(t, xa[k] - s[k], xa[k] + s[k], facecolor='b', alpha=0.3)
            ax1.set_box_aspect(1)
            ax1.set_yticks(np.linspace(-np.ceil(np.max(xa[k] + s[k])), np.ceil(np.max(xa[k] + s[k])), 21), minor=True)
            ax1.set_xticks(ticks_time)
            ax1.set_yticks(np.linspace(-np.ceil(np.max(xa[k] + s[k])), np.ceil(np.max(xa[k] + s[k])), 5))
            ax1.tick_params(which='minor', bottom=False, left=False)
            plt.ylabel('Magnitude')
            plt.ylim((-2, 2))
        else:
            plt.plot(t, np.squeeze(infotermest), linewidth=2, color='k')
            plt.xlim((0, np.max(t)))
            ax1.set_box_aspect(1)
            ax1.set_xticks(ticks_time)
            ax1.set_yticks(ticks_mi)
            ax1.tick_params(which='minor', bottom=False, left=False)
            plt.ylabel('f^-1 (I(X,Y))')
            plt.ylim(ylim1)

        plt.xlabel('Time')
        ax2 = fig.add_subplot(3, 2, 2 * (k + 1))
        if k < 2:
            if np.any(a0[k, :] > 0):
                markerline, stemlines, baseline = plt.stem(
                    f[a0[k, :] > 0], a0[k, a0[k, :] > 0], linefmt='k', markerfmt='ko', basefmt='w')
                markerline.set_markerfacecolor('none')
            if np.any(a[k, :] > 0):
                markerline, stemlines, baseline = plt.stem(
                    f[a[k, :] > 0], a[k, a[k, :] > 0], linefmt='b', markerfmt='bo', basefmt='w')
                markerline.set_markerfacecolor('none')
            plt.ylim(ylim0)
            ax2.set_box_aspect(1)
            ax2.set_yticks(ticks_pow)
        else:
            markerline, stemlines, baseline = plt.stem(
                freqs_all[r_b > 0], r_b[r_b > 0], linefmt='k', markerfmt='ko', basefmt='w')
            ax2.set_box_aspect(1)
            ax2.set_yticks(ticks_mipow)
            plt.ylim(ylim2)
            markerline.set_markerfacecolor('none')
        ax2.set_xticks(ticks_freq)
        ax2.tick_params(which='minor', bottom=False, left=False)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.show()

    return [fig]


if __name__ == '__main__':
    app.run_server(host= '0.0.0.0', debug=True)
