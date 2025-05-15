import random
import base64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import io
from cmmrt.projection.models.projector.loader import (
    _load_projector_pipeline_from
)

from dash import Dash, dcc, html, no_update, Input, Output, State, callback

# ------------------------------------------------------------------------------


def get_ppm_error(mass, ppm_error=10):
    return (round(mass) * ppm_error) / 10 ** 6

# ------------------------------------------------------------------------------


def rank_ind(train,
             data_to_rank,
             predicted,
             projector,
             plot=False):
    x, y = (
        torch.from_numpy(train.prediction.values.reshape(-1, 1)),
        torch.from_numpy(train.rt.values * 60)
    )
    projector.projector.prepare_metatesting()
    projector.fit(x, y)
    mass_error_seed = 123
    if mass_error_seed is not None:
        np.random.seed(mass_error_seed)

        candidates_list = []

    for index, row in data_to_rank.iterrows():
        # Skip if the compound is not in the test set (since it wouldn't
        # have a chance to be in the top results)
        error = get_ppm_error(row.calc_mw)
        candidates = predicted[
            (predicted["ExactMass"] >= (row.calc_mw - error))
            & (predicted["ExactMass"] <= (row.calc_mw + error))
        ].copy()

        candidates = candidates.drop(
            ['cmm_id'], axis=1)
        candidates = candidates.rename(columns={'prediction': 'rt_predicted'})

        if candidates.shape[0] > 0:
            candidates['FeatureID'] = row.FeatureID
            candidates['rt_experimental'] = row.rt * 60
            candidates['mass_experimental'] = row.calc_mw
            candidates['z_score'] = pd.NA
            candidates['mass_error'] = abs(candidates.ExactMass - row.calc_mw)
            # add small noise to unbreak ties
            candidates['mass_error'] = candidates['mass_error'] + \
                np.random.uniform(0, 1e-6, candidates.shape[0])
            candidates.sort_values(by='mass_error', inplace=True)
            scores = projector.z_score(
                candidates[['rt_predicted']].values, np.array([row.rt * 60]))
            scores = scores.cpu().numpy()
            candidates.loc[:, 'z_score'] = scores
            candidates.sort_values("z_score", inplace=True)
            candidates = candidates.nlargest(3, ['z_score'])
            candidates_list.append(candidates)

    candidates_final = pd.concat(candidates_list).reset_index(drop=True)
    candidates_final = candidates_final[['FeatureID', 'mass_experimental',
                                         'rt_experimental',
                                         'rt_predicted', 'mass_error',
                                         'z_score', 'database_id', 'Name',
                                         'MolecularFormula',
                                         'ExactMass', 'InChIKey', 'InChI']]

    # candidates_final.to_csv(candidates_file)

    # plotting
    if plot:
        sorted_x = torch.arange(
            x.min() - 0.5, x.max() + 0.5, 0.1, dtype=torch.float32)
        fig, ax = plt.subplots()
        plt.scatter(predicted.prediction.values,
                    projector.predict(predicted.prediction.values)[0])
        preds_mean, lb, ub = projector.predict(sorted_x)
        plt.scatter(x, y, marker='x')
        plt.fill_between(sorted_x, lb, ub, alpha=0.2, color='orange')
        plt.plot(sorted_x, preds_mean, color='orange')
        plt.title('Projection to database')
        plt.xlabel("Predicted RT")
        plt.ylabel("Projected/Experimental RT")
        with torch.no_grad():
            sorted_x_ = torch.from_numpy(
                projector.x_scaler.transform(sorted_x.numpy().reshape(-1, 1)))
            tmp = projector.projector.gp.mean_module(sorted_x_)
            tmp = projector.y_scaler.inverse_transform(
                tmp.reshape(-1, 1)).flatten()
            plt.plot(sorted_x, tmp, color='green')
        # plt.savefig(plot_file)
        plt.close()
        return candidates_final, fig

    return candidates


# ------------------------------------------------------------------------------
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)

    return df


# ------------------------------------------------------------------------------
def load_data(model):

    if model == 'garcia_pubchem':
        db = pd.read_csv('data/Pubchem_info_ready.csv')
        predicted = pd.read_csv('data/predicted_pubchem.csv')
    else:
        db = pd.read_csv('data/NP_atlas_info_ready.csv')
        predicted = pd.read_csv('data/predicted_NPAtlas.csv')

    db = db.astype({'database_id': 'str'})

    predicted = predicted.astype({'database_id': 'str'})
    predicted = predicted.merge(
        db, on='database_id', how='left')

    projector = _load_projector_pipeline_from(
        "data/p2e_zero_rbf+linear_4e03_-1_10",
        mean='zero',
        kernel='rbf+linear'
    )

    return predicted, projector


# Initialize the app
app = Dash(__name__)

# Adding layout
app.layout = html.Div([
    html.Div([
        html.H1('Annotate with CMM-RT', style={
            'color': '#ffffff',
            'backgroundColor': '#1f77b4',
            'padding': '20px',
            'borderRadius': '10px',
            'textAlign': 'center',
            'marginBottom': '20px'
        }),

        html.H4('Select model + database to use', style={'marginTop': '20px'}),
        dcc.Dropdown(
            options={
                'garcia_pubchem': 'Garcia, et al. (2018) model + PubChem',
                'garcia_npatlas': 'Garcia, et al. (2018) model + '
                'Natural Products Atlas'
            },
            value='garcia_pubchem',
            id='model',
            style={
                'width': '60%',
                'marginBottom': '30px'
            }
        ),

        html.Div([
            html.H4('Upload table of features to annotate'),
            dcc.Upload(
                id='df-unknowns',
                children=html.Div([
                    '📁 Drag and drop or ',
                    html.A('Select a File', style={
                           'color': '#1f77b4', 'textDecoration': 'underline'})
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderColor': '#1f77b4',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '10px 0',
                    'backgroundColor': '#f9f9f9',
                    'cursor': 'pointer'
                },
            ),
            html.Div(id='dfu-filename',
                     style={'fontStyle': 'italic', 'marginTop': '20px'})
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}
        ),

        html.Div([
            html.H4('Upload table with known compounds'),
            dcc.Upload(
                id='df-knowns',
                children=html.Div([
                    '📁 Drag and drop or ',
                    html.A('Select a File', style={
                           'color': '#1f77b4', 'textDecoration': 'underline'})
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderColor': '#1f77b4',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '10px 0',
                    'backgroundColor': '#f9f9f9',
                    'cursor': 'pointer'
                },
            ),
            html.Div(id='dfk-filename',
                     style={'fontStyle': 'italic', 'marginTop': '20px'})
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right',
                  'padding': '10px'}
        ),

        dcc.Store(id='stored-dfu'),
        dcc.Store(id='stored-dfk'),

        html.Div([
            html.Button('▶ Start Processing', id='start-button', n_clicks=0,
                        style={
                            'width': '30%',
                            'margin': '10px',
                            'backgroundColor': '#1f77b4',
                            'color': 'white',
                            'padding': '10px 20px',
                            'border': 'none',
                            'borderRadius': '5px',
                            'fontSize': '16px',
                            'cursor': 'pointer'
                        })
        ]),

        dcc.Loading(
            id="loading-spinner",
            type="circle",
            fullscreen=True,
            children=html.Div(id="processing-output"),
            style={"backgroundColor": "#ffffff80"}
        ),
        dcc.Store(id='stored-results'),
        dcc.Store(id='stored-figure'),

        html.Div(id='num-candidates'),

        html.Div([
            html.Button("🗎 Download Results", id="download-csv-btn",
                        n_clicks=0,
                        style={
                            'width': '30%',
                            'margin': '10px',
                            'backgroundColor': '#2ca02c',
                            'color': 'white',
                            'padding': '10px 20px',
                            'border': 'none',
                            'borderRadius': '5px',
                            'fontSize': '16px',
                            'cursor': 'pointer'
                        }),
            dcc.Download(id="download-csv"),

            html.Button("🎨 Download Projection Plot", id="download-img-btn",
                        n_clicks=0,
                        style={
                            'width': '30%',
                            'margin': '10px',
                            'backgroundColor': '#2ca02c',
                            'color': 'white',
                            'padding': '10px 20px',
                            'border': 'none',
                            'borderRadius': '5px',
                            'fontSize': '16px',
                            'cursor': 'pointer'
                        }),
            dcc.Download(id="download-image")
        ],
            id='download-buttons',
            style={'display': 'none'}),

        html.Div(id='processing-input')

    ], style={
        'maxWidth': '1000px',
        'margin': 'auto',
        'padding': '40px',
        'backgroundColor': '#ffffff',
        'borderRadius': '12px',
        'boxShadow': '0 4px 12px rgba(0, 0, 0, 0.1)'
    })
], style={'backgroundColor': '#f0f2f5', 'minHeight': '100vh'}
)

# Reactive logic

# Showing the filename of the uploaded files


@callback(
    Output('dfu-filename', 'children'),
    Input('df-unknowns', 'filename')
)
def update_dfu_filename(name):
    if name is not None:
        return f"Uploaded file: {name}"
    return ""


@callback(
    Output('dfk-filename', 'children'),
    Input('df-knowns', 'filename')
)
def update_dfk_filename(name):
    if name is not None:
        return f"Uploaded file: {name}"
    return ""


# Storing contents


@callback(
    Output('stored-dfu', 'data'),
    Input('df-unknowns', 'contents'),
    State('df-unknowns', 'filename')
)
def store_dfu(content, filename):
    if content:
        return {'content': content, 'filename': filename}
    return None


@callback(
    Output('stored-dfk', 'data'),
    Input('df-knowns', 'contents'),
    State('df-knowns', 'filename')
)
def store_dfk(content, filename):
    if content:
        return {'content': content, 'filename': filename}
    return None

# Getting candidates when button is pressed


@callback(Output('stored-results', 'data'),
          Output('stored-figure', 'data'),
          Output('processing-output', 'children'),
          Input('start-button', 'n_clicks'),
          State('stored-dfu', 'data'),
          State('stored-dfk', 'data'),
          State('model', 'value'))
def run_analysis(n_clicks, dfu_data, dfk_data, model):
    if n_clicks > 0 and dfu_data and dfk_data:

        predicted, projector = load_data(model)

        dfu = parse_contents(dfu_data['content'], dfu_data['filename'])
        dfk = parse_contents(dfk_data['content'], dfk_data['filename'])
        dfk = dfk.astype({'annot_id': 'str'})

        train = dfk.merge(predicted, left_on='annot_id',
                          right_on='database_id', how='inner')
        train = train.drop_duplicates()
        index_list = range(0, train.shape[0])
        ii = random.sample(index_list, 20)
        train_filt = train.iloc[ii]

        candidates_df, fig = rank_ind(train_filt,
                                      dfu,
                                      predicted,
                                      projector,
                                      plot=True)

        fig_bytes = io.BytesIO()
        if fig:
            fig.savefig(fig_bytes, format='png')
            fig_bytes.seek(0)
            fig_base64 = base64.b64encode(fig_bytes.read()).decode('utf-8')
        else:
            fig_base64 = None

        return candidates_df.to_dict('records'), fig_base64, ''
    return '', '', ''


# Show results

@callback(
    Output("num-candidates", "children"),
    Input("stored-results", "data"),
    prevent_initial_call=True
)
def num_annotated(results_data):
    if results_data:
        df = pd.DataFrame(results_data)

        n_annotated = df['FeatureID'].nunique()
        text = f"Number of annotated metabolites: {n_annotated}"

        return html.Div([html.H3('Annotation successfull',
                                 style={'color': '#2ca02c',
                                        'margin': '10px'}),
                         html.Div([text],
                                  style={'margin': '10px'})])

    return ""

# Download data

# Displaying buttons


@callback(
    Output("download-buttons", "style"),
    Input("stored-results", "data")
)
def toggle_download_buttons(results_data):
    if results_data:
        return {"display": "block"}
    return {"display": "none"}

# Candidates table


@callback(
    Output("download-csv", "data"),
    Input("download-csv-btn", "n_clicks"),
    State("stored-results", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, results_data):
    df = pd.DataFrame(results_data)
    return dcc.send_data_frame(df.to_csv, "CMMRT_results.csv", index=False)

# Projection Figure


@callback(
    Output("download-image", "data"),
    Input("download-img-btn", "n_clicks"),
    State("stored-figure", "data"),
    prevent_initial_call=True
)
def download_image(n_clicks, fig_base64):
    if fig_base64:
        fig_bytes = base64.b64decode(fig_base64)
        return dcc.send_bytes(lambda buffer: buffer.write(fig_bytes),
                              "projection_plot.png")
    return no_update


if __name__ == '__main__':
    app.run(debug=False)
