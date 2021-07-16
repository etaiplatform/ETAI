import numpy as np
from flask import Flask, request, jsonify, Response
import pandas as pd
from source import RunIterNUL
from source.Models import LGBRegression

from source.Preprocessing import preprocess_final

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def smape(A, F):
    tmp = 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    return 100 / len_ * np.nansum(tmp)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.nanmean(np.abs((y_true - y_pred) / y_true)) * 100


@app.route('/')
def home():
    return jsonify({"home": "home"})


@app.route('/prep_data', methods=['GET'])
def get_preprocessed_data():
    startDate = request.args.get('startDate')
    endDate = request.args.get('endDate')
    download = request.args.get('download')
    download = download == "True"
    data = preprocess_final(startDate, endDate)
    if download:
        return Response(
            data.to_csv(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename={}-{}.csv".format(startDate, endDate)})
    else:
        columns = data.columns.tolist()
        data = data.values.tolist()
        response = {
            "columns": columns,
            "data": data
        }
        resp = jsonify(response)
        resp.status_code = 200
        return resp


@app.route('/predict', methods=['GET'])
def predict():
    startDate = request.args.get('startDate')
    endDate = request.args.get('endDate')
    days = request.args.get('days')
    arc = request.args.get('model')
    plot = request.args.get('plot')
    predictions = []
    truth = []
    plotpath = ""
    if arc == "DEF":
        predictions, plotpath, truth = LGBRegression.start(startDate, endDate, days, plot=plot, target="consumption")
    elif arc == "BIN1":
        predictions, truth, plotpath, predictions_clf = RunIterNUL.run_binary_1_iter(startDate, endDate, days,
                                                                                     plot=plot, target="consumption")
    elif arc == "NUL1":
        predictions, truth, plotpath, predictions_clf = RunIterNUL.run_nul_1_iter(startDate, endDate, days, plot=plot,
                                                                                  target="consumption")
    elif arc == "NUL3":
        predictions, truth, plotpath, predictions_clf = RunIterNUL.run_nul_3_iter(startDate, endDate, days, plot=plot,
                                                                                  target="consumption")
        predictions = np.array(predictions["preds"].to_list())
    elif arc == "DMDNUL1":
        predictions, truth, plotpath, predictions_clf = RunIterNUL.run_dimdik(startDate, endDate, days, plot=plot,
                                                                              target="consumption")

    date_range = pd.date_range(start=pd.to_datetime(endDate) - pd.DateOffset(days=int(days) - 1),
                               end=pd.to_datetime(
                                   (pd.to_datetime(endDate) + pd.DateOffset(days=1)) - pd.DateOffset(hours=1)),
                               freq='H').tolist()
    date_range = [str(dr) for dr in date_range]
    if plotpath:
        response = {
            "date": date_range,
            "predictions": list(predictions),
            "truth": list(truth),
            "plotpath": plotpath
        }
    else:
        response = {
            "date": date_range,
            "predictions": list(predictions),
            "truth": list(truth),
        }
    resp = jsonify(response)
    resp.status_code = 200
    return resp
    # get the plotpath parameter from api


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
