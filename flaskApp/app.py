import numpy as np
from flask import Flask, request, render_template
from sklearn.metrics import mean_absolute_error, mean_squared_error
from API.Requests import predict_api

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def smape(A, F):
    A = np.array(A)
    F = np.array(F)
    tmp = 2 * np.abs(F - A) / (np.abs(A) + np.abs(F))
    len_ = np.count_nonzero(~np.isnan(tmp))
    return 100 / len_ * np.nansum(tmp)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.nanmean(np.abs((y_true - y_pred) / y_true)) * 100


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    params = [x for x in request.form.values()]
    startDate = params[0]
    endDate = params[1]
    days = params[2]
    arc = params[3]
    preds, truth, plotpath = predict_api(startDate, endDate, days, arc, plot=True)
    predictions_list = preds["predictions"].tolist()
    return render_template('index.html', path="../static/" + plotpath.split('/')[-1],
                           mae=mean_absolute_error(truth, predictions_list),
                           mape=mean_absolute_percentage_error(truth, predictions_list),
                           rmse=mean_squared_error(truth, predictions_list, squared=False),
                           smape=smape(truth, predictions_list))


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8051, debug=True)
