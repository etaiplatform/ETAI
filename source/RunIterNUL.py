from source.Models import LGBBinary, LGBNuliter, LGB_NUL_CLF, LGBRegression


# from sklearn.model_selection import RandomizedSearchCV


def run_nul_3_iter(startDate, endDate, days, plot):
    predictions_clf, data = LGB_NUL_CLF.start_multi_clf(startDate, endDate, days)
    predictions, plotpath, truth = \
        LGBNuliter.start_after_multi_models(data, predictions_clf, startDate, endDate, days, plot=plot)
    return predictions, truth, plotpath, predictions_clf


def run_nul_1_iter(startDate, endDate, days, plot):
    predictions_clf, data = LGB_NUL_CLF.start_multi_clf(startDate, endDate, days)
    predictions, plotpath, truth = LGBRegression.start_after_multi(data, predictions_clf, startDate, endDate, days,
                                                                   plot=plot)
    return predictions, truth, plotpath, predictions_clf


def run_binary_1_iter(startDate, endDate, days, plot):
    predictions_binary, data = LGBBinary.start_clf(startDate, endDate, days)
    predictions, plotpath, truth = LGBRegression.start_after(data, predictions_binary, startDate, endDate, days,
                                                             plot=plot)
    return predictions, truth, plotpath, predictions_binary


def run_dimdik(startDate, endDate, days, plot):
    predictions_clf, data = LGB_NUL_CLF.start_multi_clf(startDate, endDate, days)
    predictions, plotpath, truth = LGBRegression.start_after_dimdik_multi(data, predictions_clf, startDate, endDate,
                                                                          days,
                                                                          plot=plot)
    return predictions, truth, plotpath, predictions_clf
