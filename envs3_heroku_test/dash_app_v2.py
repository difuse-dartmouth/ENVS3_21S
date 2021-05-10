#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import geopandas as gpd
import plotly.express as px
import plotly.graph_objs as go

#import data from csv
df = pd.read_csv('https://raw.githubusercontent.com/difuse-dartmouth/21s_ENVS3/main/Module_Data/parish_data_v4.csv')
df = df.drop(columns=['FIPS', 'parish_name'])
feature_names = list(df.columns)

#this is the default linear regression figure before anything is selected from dropdown
default_fig = px.scatter(df, x="Population", y="Population",trendline='ols',trendline_color_override='red')
default_fig.update_layout(width=1000, height=750, margin={"r":0,"t":10,"l":40,"b":0})
default_fig.update_traces(marker_size=12)
default_fig.update_layout(hovermode='x')

#this is the default scatterplot figure before anything is selected from dropdown
default_fig1 = px.scatter(df, x="Population", y="Population")
default_fig1.update_layout(width=1000, height=750, margin={"r":0,"t":10,"l":40,"b":0})
default_fig1.update_traces(marker_size=12)

#create variable list and modified dataframes for dropdown and correlation matrix
variable_list2 = ['First_Covid_Wave','Second_Covid_Wave','Third_Covid_Wave','Total_Current_Covid_Deaths']
First_Covid_Wave = df.drop(columns=['Second Wave of Covid Deaths per 10k', 
                                     'Third Wave of Covid Deaths per 10k',
                                     'Current Covid Deaths per 10k thru 4/28/21'])
Second_Covid_Wave = df.drop(columns=['First Wave of Covid Deaths per 10k', 
                                      'Third Wave of Covid Deaths per 10k',
                                     'Current Covid Deaths per 10k thru 4/28/21'])
Third_Covid_Wave = df.drop(columns=['First Wave of Covid Deaths per 10k', 
                                     'Second Wave of Covid Deaths per 10k',
                                     'Current Covid Deaths per 10k thru 4/28/21'])
Total_Current_Covid_Deaths = df.drop(columns=['First Wave of Covid Deaths per 10k', 
                                               'Second Wave of Covid Deaths per 10k',
                                     'Third Wave of Covid Deaths per 10k'])
feature_names2 = list(First_Covid_Wave.columns)
std_scaler = StandardScaler()
df_std = pd.DataFrame(std_scaler.fit_transform(First_Covid_Wave), columns=First_Covid_Wave.columns)
corr = df_std.corr()
#this is the default correlation matrix figure before anything is selected from dropdown
default_fig2 = px.imshow(corr,x=feature_names2,y=feature_names2,color_continuous_scale='RdBu_r',
                labels=dict(color="Correlation Coefficient"))
default_fig2.update_layout(width=1000, height=750)
default_fig2.update_layout(width=1000, height=750, margin={"r":0,"t":10,"l":40,"b":0})

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Correlation Matrix', children=[
        html.Div([
            html.Div([
                html.Label(["Covid Wave:", dcc.Dropdown(
                    id='dropdown',
                    options=[{'label': i, 'value': i} for i in variable_list2],
                )])],
            style={'width': '49%', 'display': 'inline-block'}
                       ),
            ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'}),
        dcc.Graph(id="graph1",figure=default_fig2)]),
        dcc.Tab(label='Scatter Plot', children=[
        html.Div([
            html.Div([
                html.Label(["X Variable", dcc.Dropdown(
                    id='crossfilter-xaxis-column',
                    options=[{'label': i, 'value': i} for i in feature_names],
                )])],
            style={'width': '49%', 'display': 'inline-block'}
                       ),
            html.Div([
                html.Label(["Y Variable", dcc.Dropdown(
                    id='crossfilter-yaxis-column',
                    options=[{'label': i, 'value': i} for i in feature_names],
                 )])],
            style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),
            ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'}),
        dcc.Graph(id="graph2",figure=default_fig1)]),
        dcc.Tab(label='Linear Regression', children=[
        html.Div([
            html.Div([
                html.Label(["X Variable", dcc.Dropdown(
                    id='x_variable',
                    options=[{'label': i, 'value': i} for i in feature_names],
                )])],
            style={'width': '49%', 'display': 'inline-block'}
                       ),
            html.Div([
                html.Label(["Y Variable", dcc.Dropdown(
                    id='y_variable',
                    options=[{'label': i, 'value': i} for i in feature_names],
                 )])],
            style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),
            ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'}),
        dcc.Graph(id="graph3",figure=default_fig)]),
    ])
])

@app.callback(Output('tabs-content-classes', 'children'),
              Input('tabs-with-classes', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Tab content 1')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab content 2')
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Tab content 3')
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Tab content 4')
        ])
    
@app.callback(
    dash.dependencies.Output("graph1", "figure"), 
     [dash.dependencies.Input('dropdown', 'value')])

def filter_heatmap(dropdown):
    if dropdown == 'First_Covid_Wave':
        dropdown = First_Covid_Wave
    elif dropdown == 'Second_Covid_Wave':
        dropdown = Second_Covid_Wave
    elif dropdown == 'Third_Covid_Wave':
        dropdown = Third_Covid_Wave
    elif dropdown == 'Total_Current_Covid_Deaths':
        dropdown = Total_Current_Covid_Deaths
    feature_names = list(dropdown.columns)
    std_scaler = StandardScaler()
    df_std = pd.DataFrame(std_scaler.fit_transform(dropdown), columns=dropdown.columns)
    corr = df_std.corr()
    fig = px.imshow(corr,x=feature_names,y=feature_names,color_continuous_scale='RdBu_r',
                labels=dict(color="Correlation Coefficient"))
    fig.update_layout(width=1000, height=750)
    fig.show()

    return fig

@app.callback(
    dash.dependencies.Output("graph2", "figure"), 
     [dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
     dash.dependencies.Input('crossfilter-xaxis-column', 'value')])

def update_graph(xaxis_column_name, yaxis_column_name):

    fig = px.scatter(x=df[xaxis_column_name],y=df[yaxis_column_name])

    fig.update_xaxes(title=xaxis_column_name)

    fig.update_yaxes(title=yaxis_column_name)
    fig.update_layout(width=1000, height=750, margin={"r":0,"t":10,"l":40,"b":0})
    fig.update_traces(marker_size=12)
    
    return fig

@app.callback(
    dash.dependencies.Output("graph3", "figure"), 
     [dash.dependencies.Input('y_variable', 'value'),
     dash.dependencies.Input('x_variable', 'value')])


def update_graph(xaxis_column_name, yaxis_column_name):
    if xaxis_column_name == yaxis_column_name:
        fig = px.scatter(x=df[xaxis_column_name],y=df[yaxis_column_name], opacity=0.65, trendline='ols', 
                     trendline_color_override='red')
        fig.update_layout(width=1000, height=750, margin={"r":0,"t":10,"l":40,"b":0})
        fig.update_traces(marker_size=12)
        fig.update_xaxes(title=xaxis_column_name)
        fig.update_yaxes(title=yaxis_column_name)
        fig.update_layout(hovermode='x')
    else:
        regr = LinearRegression()
        df2 = df[[xaxis_column_name,yaxis_column_name]]
        df2 = df2.dropna(axis='rows')
        x = df2[[xaxis_column_name]].values
        y=df2[[yaxis_column_name]].values
        regr.fit(x,y)
        y_pred = regr.predict(x)
        y_inter = regr.intercept_.item()
        slope = regr.coef_.item()
        r2 = r2_score(y, y_pred)
        std_scaler = StandardScaler()
        df_std = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)
        X1 = df2[[xaxis_column_name]].values
        Y1 = df2[[yaxis_column_name]].values
        x1 = X1.flatten()
        y1 = Y1.flatten()
        data = pd.DataFrame({'x': x1, 'y': y1})
        model = ols("y ~ x", data).fit()
        p_value = model.pvalues[1]
        if p_value > 0.01:
            p_value2 = ('P > |t| = %.4f' % p_value)
        else:
            p_value2 = ('P > |t| = <0.01')
        
        fig = px.scatter(x=df[xaxis_column_name],y=df[yaxis_column_name], opacity=0.65, trendline='ols', 
                     trendline_color_override='red')
        fig.update_layout(width=1000, height=750, margin={"r":0,"t":10,"l":40,"b":0})
        fig.update_traces(marker_size=12)
        fig.update_xaxes(title=xaxis_column_name)
        fig.update_yaxes(title=yaxis_column_name)
#     fig.add_annotation(xref='paper',yref='paper', x=0.01, y=0.96,
#         text='R\N{SUPERSCRIPT TWO}= %.2f' %r2,showarrow=False,font=dict(size=15))
#     fig.add_annotation(xref='paper',yref='paper', x=0.01, y=0.999,
#         text='y = %.2f' % r2 + 'x + %.2f' % y_inter,showarrow=False,font=dict(size=15))
        fig.add_annotation(xref='paper',yref='paper', x=0.001, y=0.999,
            text=p_value2,showarrow=False,font=dict(size=18))
#     fig.add_shape(type="rect", xref="paper", yref="paper",
#                   x0=0.01, x1=0.23, y0=0.85, y1=1,
#                   fillcolor="white")
        fig.update_layout(hovermode='x')
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:




