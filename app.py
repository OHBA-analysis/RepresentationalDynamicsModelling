#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: MWJ van Es, 2022
inspired by M. Fabus (https://gitlab.com/marcoFabus/fabus2022_harmonics/-/blob/main/app.py)
"""

import plotly.graph_objs as go
from plotly.subplots import make_subplots
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
                    style={'width': '100%'},
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            style={'padding-right': '0%', 'width': '100%'},
                                            children=[
                                                html.P('Channel 1, Amplitude component 1'),
                                                dcc.Slider(
                                                    id=name[0],
                                                    min=props[0],
                                                    max=props[1],
                                                    value=props[2][0],
                                                    step=props[3],
                                                ),
                                                html.P('Channel 1, Amplitude component 2'),
                                                dcc.Slider(
                                                    id=name[1],
                                                    min=props[0],
                                                    max=props[1],
                                                    value=props[2][1],
                                                    step=props[3],
                                                ),
                                                html.P('Channel 2, Amplitude component 1'),
                                                dcc.Slider(
                                                    id=name[2],
                                                    min=props[0],
                                                    max=props[1],
                                                    value=props[2][2],
                                                    step=props[3],
                                                ),
                                                html.P('Channel 2, Amplitude component 2'),
                                                dcc.Slider(
                                                    id=name[3],
                                                    min=props[0],
                                                    max=props[1],
                                                    value=props[2][3],
                                                    step=props[3],
                                                ), ])
                                    ],
                                )
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
                    style={'width': '100%'},
                    children=[

                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            style={'padding-right': '0%', 'width': '100%'},
                                            children=[
                                                html.P('Frequency component 1'),
                                                dcc.Slider(
                                                    id=name[0],
                                                    min=props[0],
                                                    max=props[1],
                                                    value=props[2][0],
                                                    step=props[3],
                                                ),
                                                html.P('Frequency component 2'),
                                                dcc.Slider(
                                                    id=name[1],
                                                    min=props[0],
                                                    max=props[1],
                                                    value=props[2][1],
                                                    step=props[3],
                                                ), ])

                                    ],
                                )
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
                {"label": "Example 1", "value": 1},
                {"label": "Example 2", "value": 2},
            ],
            value=1,
            id="example_button",
            inline=True,
        ),
    ]
)

# %% Example plotly figure
headers = ['Frequency (Hz)', 'Condition 1', 'Condition 2']
names = [str(i) for i in range(10)]

marks_amp = {str(x): {'label': str(round(x, 2)), 'style': {'color': 'black'}} for x in np.linspace(0, 2, 9)}
marks_f = {str(int(x)): {'label': str(round(x)), 'style': {'color': 'black'}} for x in np.linspace(0, 20, 5)}

props_amp = [0, 2, [1.5, 0, 0.75, 0], 0.25, marks_amp]
props_f = [0, 20, [10, 0], 1, marks_f]

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
server=app.server
app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        html.H1('Representational Dynamics Simulator'),

        dbc.Row(
            [
                dbc.Col([
                    dbc.Col(children=[instructions, radioitems]),

                    dbc.Col(dbc.Card(card_freq(headers[0], names[:2], props_f), color="light",
                                     inverse=False)),
                ]),
                dbc.Col(dbc.Card(card_amp(headers[1], names[2:6], props_amp), color="light",
                                 inverse=False)),
                dbc.Col(dbc.Card(card_amp(headers[2], names[6:10], props_amp), color="light",
                                 inverse=False))
            ], style={"width": "100vw"}),
        dbc.Row(
            [dcc.Graph(id='graph'),
             ],
        ),
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
     Output('example_button', component_property='value'),
     Output('0', component_property='value'),
     Output('1', component_property='value'),
     Output('2', component_property='value'),
     Output('3', component_property='value'),
     Output('4', component_property='value'),
     Output('5', component_property='value'),
     Output('6', component_property='value'),
     Output('7', component_property='value'),
     Output('8', component_property='value'),
     Output('9', component_property='value')],
    [Input(component_id='example_button', component_property='value')] +
    [Input(x, 'value') for x in names], )

def update_figure(example, f1, f2, s1ch1f1, s1ch1f2, s1ch2f1, s1ch2f2, s2ch1f1, s2ch1f2, s2ch2f1,
                  s2ch2f2):
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

        ylim1 = (0, 1.6)
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
        ticks_amp = np.linspace(int(np.floor(np.min((np.min(xa[0]) - s[0], np.min(xa[1]) - s[1])))),
                                int(np.ceil(np.max((np.max(xa[0]) + s[0], np.max(xa[1]) + s[1])))),
                                int(np.diff((np.floor(np.min((np.min(xa[0]) - s[0], np.min(xa[1]) - s[1]))),
                                             np.ceil(np.max((np.max(xa[0]) + s[0], np.max(xa[1]) + s[1])))))[0] + 1))
        ticks_pow = np.linspace(0, np.max((np.max(a), np.max(a0))), int(2 * np.max((np.max(a), np.max(a0))) + 1))
        ticks_mi = np.linspace(0, np.ceil(np.max(infotermest)), int(2 * np.ceil(np.max(infotermest)) + 1))
        ticks_mipow = np.linspace(0, np.ceil(np.max(r_b)), int(2 * np.ceil(np.max(r_b)) + 1))
        ylim1 = (0, np.ceil(np.max(infotermest)))
        ylim2 = (0, np.ceil(np.max(r_b)))

    example = 0

    # %% Plot everything
    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=("Stimulus Evoked Signals", "Power", "", "", "Information Content", ""))
    # left side of the figure
    for k in range(3):
        if k < 2:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=x0[k],
                    line=dict(color='black'),
                    mode='lines',
                    showlegend=False
                ),
                row=k + 1,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=t.tolist() + t.tolist()[::-1],  # x, then x reversed
                    y=(x0[k] + s[k]).tolist() + (x0[k] - s[k])[::-1].tolist(),  # upper, then lower reversed
                    fill='toself',
                    fillcolor='gray',
                    opacity=0.3,
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=k + 1,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=xa[k],
                    line=dict(color='blue'),
                    mode='lines',
                    showlegend=False
                ),
                row=k + 1,
                col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=t.tolist() + t.tolist()[::-1],  # x, then x reversed
                    y=(xa[k] + s[k]).tolist() + (xa[k] - s[k])[::-1].tolist(),  # upper, then lower reversed
                    fill='toself',
                    fillcolor='blue',
                    opacity=0.3,
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ),
                row=k + 1,
                col=1
            )
            # fig.update_layout(xaxis=dict(title_text="Time (s)"), yaxis=dict(title_text="Magnitude"))
            fig.update_yaxes(dict(title_text=f"CHANNEL {k + 1} <br> Magnitude"), range=[-2, 2], row=k + 1, col=1)
        else:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=infotermest,
                    line=dict(color='black'),
                    mode='lines',
                    showlegend=False
                ),
                row=k + 1,
                col=1
            )
            fig.update_xaxes(dict(title_text="Time (s)"), row=k + 1, col=1)
            fig.update_yaxes(dict(title_text="f<sup>-1</sup> (I<sub>(X,Y)</sub>)"), range=ylim1, row=k + 1, col=1)

        # right side of the figure
        if k < 2:
            for i in np.where(a0[k, :] > 0)[0]:
                if f[i] > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=[f[i], f[i]],
                            y=[-1, a0[k, i]],
                            mode='lines+markers',
                            line=dict(color='black', width=3),
                            marker=dict(size=10, color='black', symbol='circle'),
                            showlegend=False
                        ),
                        row=k + 1,
                        col=2
                    )
            for i in np.where(a[k, :] > 0)[0]:
                if f[i] > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=[f[i], f[i]],
                            y=[-1, a[k, i]],
                            mode='lines+markers',
                            line=dict(color='blue', width=3),
                            marker=dict(size=10, color='blue', symbol='circle'),
                            showlegend=False
                        ),
                        row=k + 1,
                        col=2
                    )
            fig.update_yaxes(dict(title_text=f"PSD"), range=ylim0, row=k + 1, col=2)
            fig.update_xaxes(range=[0, 40], row=k + 1, col=2)
        else:
            for i in np.where(r_b > 0)[0]:
                if freqs_all[i] > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=[freqs_all[i], freqs_all[i]],
                            y=[-1, r_b[i]],
                            mode='lines+markers',
                            line=dict(color='black', width=3),
                            marker=dict(size=10, color='black', symbol='circle'),
                            showlegend=False
                        ),
                        row=k + 1,
                        col=2
                    )
                fig.update_xaxes(dict(title_text="Frequency (Hz)"), range=[0, 40], row=k + 1, col=2)
                fig.update_yaxes(dict(title_text="PSD"), range=ylim2, row=k + 1, col=2)

    return [fig, example, f1, f2, s1ch1f1, s1ch1f2, s1ch2f1, s1ch2f2, s2ch1f1, s2ch1f2, s2ch2f1, s2ch2f2]


if __name__ == '__main__':
    #app.run_server(host='0.0.0.0', debug=True)
    app.debug=True
    app.run()
