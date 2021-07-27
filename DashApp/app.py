import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
import datetime
# from flask_caching import Cache
# import os
from sklearn.metrics import mean_absolute_percentage_error

from ETAI.API.Requests import get_prep_data, predict_api

df = get_prep_data("2018-01-01", "2021-06-05", "False")
preds = None
TEMPLATE = 'plotly_white'
colorsIdx = {1: 'red', 0: 'grey', -1: 'green'}
cols = df['target'].map(colorsIdx)
app = dash.Dash(
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True
)
server = app.server
# CACHE_CONFIG = {
#     # try 'filesystem' if you don't want to setup redis
#     'CACHE_TYPE': 'redis',
#     'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
# }
# cache = Cache()
# cache.init_app(app.server, config=CACHE_CONFIG)
"""Homepage"""
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])

# index_page = html.Div([
#     html.Br(),
#     html.Br(),
#     dbc.Row([
#         dbc.Col(html.H1(children="Welcome to Peaky Finders"), width=5),
#         dbc.Col(width=5),
#     ], justify='center'),
#     html.Br(),
#     html.Br(),
#     dbc.Row([
#         dbc.Col(
#             html.Div([
#                 html.H4(
#                     children="To what extent do weather and weekday determine total electricity demand on the grid? Click an ISO button below to find out."),
#                 html.Div(
#                     [
#                         dcc.Link(
#                             html.Button('HOME', id='home-button', className="mr-1"),
#                             href='/'),
#                         dcc.Link(
#                             html.Button('MAIN', id='nyiso-button', className="mr-1"),
#                             href='/nyiso'),
#                     ]
#                 )]), width=7
#         ),
#         dbc.Col(width=3),
#     ], justify="center"),
#     html.Br(),
#     html.Br(),
#     html.Br(),
#     html.Br(),
#     dbc.Row([
#         dbc.Col(html.H4(children="ISO Territory Map"), width=4),
#         dbc.Col(width=4)
#     ], justify='center'),
#     html.Div([
#         dcc.Graph(figure=px.bar(
#             x=df['year'],
#             y=df['dayAheadPrices'],
#             color=df['year'],
#             template='plotly_dark'
#         ))
#     ], style={'display': 'inline-block', 'width': '90%'})
# ])

"""NYISO LAYOUT"""
nyiso_layout = html.Div([
    html.Div(id='nyiso-content'),
    html.Br(),
    dbc.Row([
        dbc.Col(
            html.Div(
                [
                    # dcc.Link(
                    #     html.Button('HOME', id='home-button', className="mr-1"),
                    #     href='/nyiso'),
                    dcc.Link(
                        html.Button('NYISO', id='nyiso-button', className="mr-1"),
                        href='/nyiso'),
                ]
            ), width=4),
        dbc.Col(width=7),
    ], justify='center'),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H1('Turkish Electricity Market DayAhead Price Prediction'), width=9),
        dbc.Col(width=2),
    ], justify='center'),
    dbc.Row([
        dbc.Col(
            html.Div(children='''
            "The EPIAS is the Turkey's Independent System Operator — the organization responsible for
            managing Turkey’s electric grid and its competitive wholesale electric marketplace." For more information,
            visit https://www.epias.com/.
        '''), width=9),
        dbc.Col(width=2)
    ], justify='center'),
    html.Br(),
    dbc.Row([
        dbc.Col(
            html.H3('Model Training'), width=9
        ),
        dbc.Col(width=2),
    ], justify='center'),
    html.Br(),
    dbc.Row([
        dbc.Col(
            dcc.DatePickerRange(
                id='my-date-picker-range',
                min_date_allowed=date(2016, 1, 1),
                max_date_allowed=date(2022, 1, 1),
                start_date=date(2016, 1, 1),
                end_date=date.today(),
                initial_visible_month=date.today()
            ), width=3
        ),
        dbc.Col(
            dcc.Dropdown(
                id='model-select-dropdown',
                options=[
                    {'label': 'Upper-Lower-Normal 3 Models', 'value': 'NUL3'},
                    {'label': 'Upper-Lower-Normal 1 Model', 'value': 'NUL1'},
                    {'label': 'Spike-Normal 1 Model', 'value': 'BIN1'},
                    {'label': 'Normal 1 model', 'value': 'DEF'},
                    {'label': 'DMDNUL1', 'value': 'DMDNUL1'},
                ],
                multi=False,
            ), width=2
        ),
        dbc.Col(
            dcc.Dropdown(
                id='target-select-dropdown',
                options=[
                    {'label': 'price', 'value': 'price'},
                    {'label': 'consumption', 'value': 'consumption'},
                    {'label': 'production', 'value': 'production'},
                ],
                multi=False,
            ), width=2
        ),
        dbc.Col(
            dbc.Input(id="model-predict-days", type='number', placeholder='number of days to predict',
                      style={"background-color": "#EEE5E3"}), width=2
        ),
        dbc.Col(
            dbc.Button("Train Model", color="dark", className="mr-1", id='model-train-button')
            , width=2),
    ], justify='center'),
    html.Div(id='output-modle'),
    html.Br(),
    dbc.Row([
        dbc.Col(
            html.H3('Model Performance'), width=9
        ),
        dbc.Col(width=2),
    ], justify='center'),
    dbc.Row([
        dbc.Col(
            html.Div(
                id="model-predict-metric",
            ), width=9
        ),
        dbc.Col(width=2),
    ], justify='center'),
    html.Br(),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='nyiso-dropdown',
                options=[
                    {'label': 'Actual', 'value': 'Actual'},
                    {'label': 'Predicted', 'value': 'Predicted'}
                ],
                value=['Actual', 'Predicted'],
                multi=True,
            ), width=6
        ),
        dbc.Col(width=5),
    ], justify='center'),
    html.Div([
        dcc.Graph(id='nyiso-graph'),
        dbc.Col(
            dbc.Button("Update Plot", color="dark", className="mr-1", id='update-plot-button')
            , width=2),

        # dcc.Interval(
        #             id='interval-component',
        #             interval=60*1000, # in milliseconds
        #             n_intervals=0
        #         )
    ]),

    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col(html.H3('Training Data'), width=9),
        dbc.Col(width=2)
    ], justify='center'),
    dbc.Row([
        dbc.Col(
            html.Div(children='''
                    The forecasting model was trained only on historical price data
                    from 2016-2021. It is a 2 model pipeline with a classifier.
                '''), width=9
        ),
        dbc.Col(width=2)
    ], justify='center'),
    html.Br(),
    dbc.Row(
        [
            dbc.Col(
                html.Div([
                    dcc.Graph(
                        figure=px.histogram(
                            df['dayAheadPrices'],
                            x='dayAheadPrices',
                            nbins=75,
                            marginal="rug",
                            title=f"Distribution of EPIAS day ahead prices' daily peaks",
                            color_discrete_sequence=['darkturquoise']
                        ).update_layout(
                            template=TEMPLATE,
                            xaxis_title='Peak Prices (TRY)',
                            yaxis_title='Number of Days'
                        )
                    ),
                ]), width=4),
            dbc.Col(
                html.Div([
                    dcc.Graph(
                        figure=go.Figure().add_trace(
                            go.Scatter(
                                x=df['dayAheadPrices'].reset_index().index,
                                y=df['dayAheadPrices'].values,
                                mode='lines+markers',
                                marker=dict(size=3, color=cols)
                            )).update_layout(
                            title="Peak Prices Colored by their targets (lower, upper, non) spike",
                            xaxis_title="Number of Days",
                            yaxis_title="Prices (TRY)",
                            template=TEMPLATE),
                    ),
                ]), width=4),
            dbc.Col(
                html.Div([
                    dcc.Dropdown(
                        id='nyiso-scatter-dropdown',
                        options=[
                            {'label': 'year', 'value': 'year'},
                            {'label': 'month', 'value': 'month'}
                        ],
                        value='year',
                        multi=False,
                    ),
                    dcc.Graph(id='nyiso-scatter')
                ]
                ), width=4),
        ]
    ),
    # dcc.Store(id='signal')
])


#
# @cache.memoize()
# def global_store(value):
#     # simulate expensive query
#     predictions, truth = predict_api("2016-01-01", datetime.datetime.today().date(), 2, "NUL1", plot=False, target="price")
#     print('Computing value with {}'.format(value))
#     return [predictions, truth]
#
# @app.callback(dash.dependencies.Output('signal', 'data'), dash.dependencies.Input('dropdown', 'value'))
# def get_todays_predictions(value):
#     # compute value and send a signal when done
#     global_store(value)
#     return value
# @app.callback(dash.dependencies.Output('nyiso-content', 'children'),
#               [dash.dependencies.Input('nyiso-button', 'value')])

@app.callback(
    dash.dependencies.Output('model-predict-metric', 'children'),
    [dash.dependencies.Input('model-train-button', 'n_clicks')],
    state=[
        dash.dependencies.State('target-select-dropdown', 'value'),
        dash.dependencies.State('model-select-dropdown', 'value'),
        dash.dependencies.State('my-date-picker-range', 'start_date'),
        dash.dependencies.State('my-date-picker-range', 'end_date'),
        dash.dependencies.State('model-predict-days', 'value'),
    ]
)
def run_model(n_clicks, target_select, model_select, start_date, end_date, days):
    global df
    global preds
    # fig = go.Figure()
    # truth = None
    # predictions = pd.DataFrame()
    # predictions.to_csv(start_date + 'preds' + start_date + str(days) + '.csv', index=False)
    if not n_clicks:
        return '''Waiting for the model to be trained'''
    if not days:
        days = 1
        model_select = "NUL1"
        target_select = "price"
    _, truth = predict_api(start_date, end_date, days, model_select, plot=False, target=target_select)
    preds = _
    del _
    # predictions = pd.DataFrame(predictions, columns=["predictions"])
    # preds = predictions
    mae = mean_absolute_percentage_error(truth, preds["predictions"].tolist())
    return '''Mean Absolute Error (MAE) for {}-{}: {:.1%} '''.format(start_date, end_date, mae)

    # return render_template('index.html', path=plotpath, mae=mean_absolute_error(truth, predictions),
    #                        mape=mean_absolute_percentage_error(truth, predictions),
    #                        rmse=mean_squared_error(truth, predictions, squared=False), smape=smape(truth, predictions))
    # return fig.update_layout(
    #     title="DayAhead Price: Actual vs. Predicted",
    #     xaxis_title="Date",
    #     yaxis_title="Price (TRY)",
    #     template=TEMPLATE


# def compute(n_clicks, model_select, start_date, end_date):
#     return 'A computation based off of {}, {}, {}, {}'.format(
#         model_select, start_date, end_date, days
#     )

@app.callback(dash.dependencies.Output('nyiso-graph', 'figure'),
              [dash.dependencies.Input('nyiso-dropdown', 'value'),
               dash.dependencies.Input('update-plot-button', 'n_clicks')],
              state=[
                  dash.dependencies.State('model-predict-days', 'value'),
              ])
# dash.dependencies.Input('interval-component', 'n_intervals')
def plot_nyiso_load_(value, n_clicks, n_days):
    # if n_days and df.empty else[]
    # if n_days and df.empty else[]
    # if n_days and preds else[]
    # if n_days and preds else[]
    print(n_days, "???")
    global preds

    fig = go.Figure()
    if not n_days:
        return fig
    if 'Actual' in value:
        fig.add_trace(go.Scatter(
            x=list(range(len(df.iloc[-int(n_days) * 24:]['dayAheadPrices']))),
            y=list(df.iloc[-int(n_days) * 24:]['dayAheadPrices'].values),
            name='Actual Load',
            line=dict(color='maroon', width=3)))
    if 'Predicted' in value:
        fig.add_trace(go.Scatter(
            x=list(
                preds.iloc[-int(n_days) * 24:][
                    "predictions"].reset_index().index.to_list()),
            y=list(preds.iloc[-int(n_days) * 24:]["predictions"].values),
            name='Forecasted Load',
            line=dict(color='darkturquoise', width=3, dash='dash')))
    return fig.update_layout(
        title="DayAhead Price: Actual vs. Predicted",
        xaxis_title="Date",
        yaxis_title="Price (TRY)",
        template=TEMPLATE
    )


@app.callback(dash.dependencies.Output("nyiso-scatter", "figure"),
              [dash.dependencies.Input("nyiso-scatter-dropdown", "value")])
def nyiso_scatter_plot(value):
    dff = df.groupby(value).mean()[["dayAheadPrices"]].reset_index()
    fig = px.bar(
        data_frame=dff,
        x=value,
        y='dayAheadPrices',
        color=value,
        #         template='plotly_dark'
    )
    return fig.update_layout(template=TEMPLATE, title='Average price over the ' + value + 's')


# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/nyiso':
        return nyiso_layout
    else:
        return nyiso_layout


if __name__ == "__main__":
    app.run_server(host='0.0.0.0', debug=False)
