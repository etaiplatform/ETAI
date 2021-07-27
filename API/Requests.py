import requests, json
import pandas as pd

host = "http://0.0.0.0:5000"


def get_prep_data(startDate, endDate, download):
    url = "prep_data"
    req = "{}/{}?startDate={}&endDate={}&download={}".format(host, url, startDate, endDate, download)
    res = requests.request("get", req)
    if download == "False":
        json_data = json.loads(res.text.encode('utf8'))
        df = pd.DataFrame(json_data["data"], columns=json_data["columns"])
        return df
    return res


def predict_api(startDate, endDate, days, model, plot, target):
    url = "predict"
    req = "{}/{}?startDate={}&endDate={}&days={}&model={}&plot={}&target={}".format(host, url, startDate, endDate, days,
                                                                                    model,
                                                                                    plot, target)
    print(req)
    res = requests.request("get", req)
    json_data = json.loads(res.text.encode('utf8'))
    if plot == "True" or plot == True:
        return pd.DataFrame(json_data["predictions"], columns=["predictions"]), \
               json_data["truth"], json_data["plotpath"]
    else:
        return pd.DataFrame(json_data["predictions"], columns=["predictions"]), \
               json_data["truth"]
