# importing libraries
from __future__ import division, print_function
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_log_error
from scipy.stats import linregress


def process_modelParams(df):
    """function to process the model parameters

    Args:
        df (dataFrame): dataFrame of the model parameters

    Returns:
        dataFrame: Modified dataFrame of the model parameters
    """
    # df['Rrs547'] = df[['Rrs550', 'Rrs555']].mean(axis=1)
    # df['Rrs667'] = df[['Rrs665', 'Rrs670']].mean(axis=1)
    df["adg443"] = df["ad443"] + df["ag443"]
    df["apg443"] = df["ap443"] + df["ag443"]
    df["R(aph/adg)"] = df["aph443"] / df["adg443"]
    df["R(ad/ag)"] = df["ad443"] / df["ag443"]
    df["bbp443"] = df["bb443"] - 0.0024484
    # 'chl'
    return df


def process_insituData(fname):
    """function to process the insitu dataset

    Args:
        fname (str): path of the insitu dataset

    Returns:
        dataFrame: dataFrame of the processed insitu dataset
    """
    data = pd.read_csv(fname)
    data = data.replace(-999.0, np.nan)
    data["lw530"] = data["lw530"].replace(np.nan, -999.0)
    data["es530"] = data["es530"].replace(np.nan, -999.0)
    Bands = [411, 443, 489, 510, 520, 530, 550, 555, 665, 670]
    for num in Bands:
        data["lw" + str(num)].fillna((data["lw" + str(num)].mean()), inplace=True)
        data["es" + str(num)].fillna((data["es" + str(num)].mean()), inplace=True)
        data["Rrs" + str(num)] = (
            data["lw" + str(num)] / data["es" + str(num)]
        ) * 0.54  # Correction Factor = *0.54

    data["Rrs530"].loc[data["Rrs530"] > 0.99999] = np.nan
    data["Rrs530"].loc[data["Rrs530"] < 0.0] = np.nan
    data["Rrs531"] = data.apply(
        lambda x: (x.Rrs510 + x.Rrs550) / 2 if np.isnan(x.Rrs530) else x.Rrs530, axis=1
    )
    data["aph443"] = data["ap443"] - data["ad443"]

    return data


def processing_syntheticData(fname):
    """function to process the synthetic dataset

    Args:
        fname (str): path of the synthetic dataset

    Returns:
        dataFrame: dataFrame of the processed synthetic dataset
    """
    XLS1 = pd.ExcelFile(fname)
    sheets = ["a_ph", "a_g", "a_dm", "a", "bb", "Rrs"]
    cols = ["aph443", "ag443", "ad443", "a443", "bb443"]
    synth_data = pd.DataFrame()
    for s in range(len(sheets)):
        temp1 = XLS1.parse(sheets[s], skiprows=8)
        temp1.drop(temp1.columns[[0]], axis=1, inplace=True)
        temp1.columns = temp1.columns.astype(str)
        if s < len(cols):
            synth_data[cols[s]] = temp1["440"]
        else:
            # temp1.rename(columns={'410':'411', '440':'443','490':'489','530':'531','560':'555','660':'665','680':'683'}, inplace=True)
            temp1.rename(
                columns={
                    "410": "410",
                    "440": "440",
                    "490": "490",
                    "530": "530",
                    "560": "560",
                    "660": "660",
                    "680": "680",
                },
                inplace=True,
            )
            temp1 = temp1.add_prefix("Rrs")
            synth_data = pd.concat([temp1, synth_data], axis=1)
    synth_data["ap443"] = synth_data["aph443"] + synth_data["ad443"]

    return synth_data


def build_model(train_features, opt, loss, activation="relu", reg=None, out=2):
    """function to build the model

    Args:
        train_features (dataFrame): dataFrame of the training features
        opt (str): Optimizer
        loss (str): Loss function
        activation (str, optional): Activation function. Defaults to "relu".
        reg (Regularizer Object, optional): Regularization. Defaults to None.
        out (int, optional): Output dimension. Defaults to 2.

    Returns:
        Model: Model Object
    """
    model = Sequential(
        [
            Dense(
                6,
                activation=activation,
                input_shape=[len(train_features[0])],
                kernel_regularizer=reg,
            ),
            Dense(out, activation="linear"),
        ]
    )

    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
    return model


def plotter(history, model_name):
    """function to plot the train loss

    Args:
        history (dict): history of the model
        model_name (str): name of the model
    """
    # mae=history.history['val_loss']
    loss = history.history["loss"]

    epochs = range(len(loss))  # Get number of epochs

    # plt.plot(epochs, mae, 'r')
    plt.plot(epochs, loss, "b")
    # plt.title('Val Loss and Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss"])

    plt.figure()
    plt.savefig("./results/loss" + model_name + ".png")


def data_processing(synthetic_datasets, insitu_dataset, Rrs):
    """function to process the data

    Args:
        synthetic_datasets (dataFrame): dataFrame of the synthetic dataset
        insitu_dataset (dataFrame): dataFrame of the insitu dataset
        Rrs (list): Parameters to be used for training

    Returns:
        dataFrame: dataFrame of the processed data
    """
    # Synthetic IOCCG 2005 dataset
    synth_data = pd.concat(
        [
            processing_syntheticData(synthetic_datasets[0]),
            processing_syntheticData(synthetic_datasets[1]),
        ],
        axis=0,
    )
    print("Synthetic IOCCG 2005 dataset: ", synth_data)

    # NOMAD Insitu dataset
    # insitu_data = process_insituData('./data/NOMAD_DATA.csv')
    # print("NOMAD Insitu dataset: ", insitu_data)

    # I/O selection
    data = process_modelParams(data)
    req_columns = Rrs + [
        "a443",
        "ap443",
        "ag443",
        "ad443",
        "aph443",
        "apg443",
        "adg443",
        "bbp443",
        "R(ad/ag)",
        "R(aph/adg)",
    ]
    data = data[req_columns]
    data.dropna(inplace=True)
    # Adding 10% Noise to Inputs.
    # data[Rrs] = data[Rrs]+np.random.uniform(-1,1,data[Rrs].shape)
    # data[Rrs] = data[Rrs]+data[Rrs]*(0.10)
    print("Input/Output dataset: ", data)
    return data


def feature_processing(data, Rrs):
    """function to process the features
    
    Args:
        data (dataFrame): dataFrame of the dataset
        Rrs (list): Parameters to be used for training
    
    Return:
        dataFrame: dataFrame of the processed features
    """
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train = train.apply(lambda x: np.log10(x, out=np.zeros_like(x), where=(x != 0)))
    test = test.apply(lambda x: np.log10(x, out=np.zeros_like(x), where=(x != 0)))

    X_train = train[Rrs]

    Y_train1 = train[["apg443", "bbp443"]]
    Y_train2 = train[["R(aph/adg)"]]
    Y_train3 = train[["R(ad/ag)"]]
    X_test = test[Rrs]
    Y_test1 = test[["apg443", "bbp443"]]
    Y_test2 = test[["R(aph/adg)"]]
    Y_test3 = test[["R(ad/ag)"]]
    # define standard scaler for normalizing input data
    X_train_scaler = StandardScaler()
    Y_train1_scaler = StandardScaler()
    Y_train2_scaler = StandardScaler()
    Y_train3_scaler = StandardScaler()
    # transform training data
    X_train = X_train_scaler.fit_transform(X_train)
    Y_train1 = Y_train1_scaler.fit_transform(Y_train1)
    Y_train2 = Y_train2_scaler.fit_transform(Y_train2)
    Y_train3 = Y_train3_scaler.fit_transform(Y_train3)
    # transform testing data
    X_test = X_train_scaler.transform(X_test)
    Y_test1 = Y_train1_scaler.transform(Y_test1)
    Y_test2 = Y_train2_scaler.transform(Y_test2)
    Y_test3 = Y_train3_scaler.transform(Y_test3)

    print("Training data X (" + " ".join(Rrs) + "):", X_train.shape)
    print("Training data Y1 (apg443, bbp443):", Y_train1.shape)
    print("Training data Y2 (R[aph/adg]):", Y_train2.shape)
    print("Training data Y3 (R[ad/ag]):", Y_train3.shape)
    print("Testing data X:", X_test.shape)
    print("Testing data Y1:", Y_test1.shape)
    print("Testing data Y2:", Y_test2.shape)
    print("Testing data Y3:", Y_test3.shape)

    return (
        X_train,
        Y_train1,
        Y_train2,
        Y_train3,
        X_test,
        Y_test1,
        Y_test2,
        Y_test3,
        X_train_scaler,
        Y_train1_scaler,
        Y_train2_scaler,
        Y_train3_scaler,
        test,
    )


def model_results(model, X_test, Y_test):
    """function to evaluate the model

    Args:
        model (object): Model object
        X_test (numpy array): Testing data features
        Y_test (numpy array): Testing data labels

    Returns:
        numpy array: Predicted values 
    """
    predicted = model.predict(X_test)
    results = model.evaluate(X_test, Y_test, batch_size=32)
    print(
        " ",
        " ".join(model.metrics_names),
        "\n",
        " ".join(map(str, [round(item, 5) for item in results])),
    )
    return predicted


def prediction_plotter(Y_test, predicted, model_name, param):
    """function to plot the predicted values

    Args:
        Y_test (numpy array): Testing data labels
        predicted (numpy array): Predicted values 
        model_name (str): Name of the model 
        param (str): Title of the plot
    """
    fig, ax = plt.subplots()
    y = Y_test
    pred = predicted
    ax.scatter(y, pred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k-", lw=2)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.title(param)
    plt.show()
    plt.savefig("./results/" + model_name + "_" + param + ".png")


def iop_product_plotter(
    rmse, slope, param_name, org_param, pred_param, model_name, param_pos, r2
):
    """function to plot the iop product values

    Args:
        rmse (int): Root mean square error 
        slope (int): Slope of the line  
        param_name (str): Name of the parameter 
        org_param (numpy array): original iop product values
        pred_param (numpy array): predicted iop product values
        model_name (str): Name of the model 
        param_pos (list): position for each parameter on the plot
        r2 (int): R^2 value 
    """
    fig, ax = plt.subplots()
    y = org_param
    pred = pred_param
    ax.scatter(y, pred)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k-", lw=2)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    ax.text(0, param_pos[0], "R^2 = " + str(round(r2, 2)))
    ax.text(0, param_pos[1], "RMSE = " + str(round(rmse, 2)))
    ax.text(0, param_pos[2], "Linear % error = " + str(round(10 ** (rmse) - 1, 2)))
    ax.text(0, param_pos[3], "Slope = " + str(round(slope, 2)))
    # ax.text(0, param_pos[4], 'Intercept = ' + str(round(inter ,2)))
    ax.text(0, param_pos[4], "N = " + str(len(y)))
    plt.title(param_name)
    plt.show()
    plt.savefig("./results/" + model_name + "_" + param_name + ".png")


def iop_product_statistics(
    org_param1,
    org_param2,
    pred_param1,
    pred_param2,
    model_name,
    param1_name,
    param2_name,
    param1_pos,
    param2_pos,
):
    """function to calculate the statistics for the iop product values

    Args:
        org_param1 (numpy array): original iop product values for param1
        org_param2 (numpy array): original iop product values for param2
        pred_param1 (numpy array): predicted iop product values for param1
        pred_param2 (numpy array): predicted iop product values for param2
        model_name (str): Name of the model 
        param1_name (str): Name of the parameter 1
        param2_name (str): Name of the parameter 2 
        param1_pos (list): position for each parameter 1 on the plot
        param2_pos (list): position for each parameter 2 on the plot
    """
    r2_param1 = r2_score(pred_param1, org_param1)
    r2_param2 = r2_score(pred_param2, org_param2)
    print(param1_name + " R2 score: ", r2_param1)
    print(param2_name + " R2 score: ", r2_param2)

    rmse_param1 = np.sqrt(mean_squared_log_error(pred_param1, org_param1))
    rmse_param2 = np.sqrt(mean_squared_log_error(pred_param2, org_param2))
    LR_param1 = linregress(pred_param1, org_param1)
    LR_param2 = linregress(pred_param2, org_param2)
    slope_param1 = LR_param1.slope
    inter_param1 = LR_param1.intercept
    slope_param2 = LR_param2.slope
    inter_param2 = LR_param2.intercept

    iop_product_plotter(
        rmse_param1,
        slope_param1,
        param1_name,
        org_param1,
        pred_param1,
        model_name,
        param1_pos,
        r2_param1,
    )
    iop_product_plotter(
        rmse_param2,
        slope_param2,
        param2_name,
        org_param2,
        pred_param2,
        model_name,
        param2_pos,
        r2_param2,
    )


def main():
    """
        Main function
    """
    # data pre-processing
    # 'Rrs411', 'Rrs443', 'Rrs489', 'Rrs531', 'Rrs547', 'Rrs667', 'Rrs683', Rrs690', 'Rrs700','Rrs710'
    # 410, 440, 490, 510, 550, 620, 670, 680, 710
    # Rrs_1 = ['Rrs410', 'Rrs440', 'Rrs490', 'Rrs510', 'Rrs550']
    Rrs = [
        "Rrs410",
        "Rrs440",
        "Rrs490",
        "Rrs510",
        "Rrs550",
        "Rrs620",
        "Rrs670",
        "Rrs680",
        "Rrs710",
    ]
    data = data_processing(
        ["./data/IOP_AOP_Sun30.xls", "./data/IOP_AOP_Sun30.xls"],
        "./data/NOMAD_DATA.csv",
        Rrs,
    )
    # training and testing data split & feature processing

    (
        X_train,
        X_test,
        Y_train1,
        Y_test1,
        Y_train2,
        Y_test2,
        Y_train3,
        Y_test3,
        X_train_scaler,
        Y_train1_scaler,
        Y_train2_scaler,
        Y_train3_scaler,
        test,
    ) = feature_processing(data, Rrs)

    # model training
    model1 = build_model(X_train, "Adam", "mse", "tanh", 2)
    history = model1.fit(X_train, Y_train1, batch_size=32, epochs=600, verbose=2)
    plotter(history, "model1")

    model2 = build_model(X_train, "Adam", "mse", "tanh", 1)
    history = model2.fit(X_train, Y_train2, batch_size=32, epochs=1000, verbose=2)
    plotter(history, "model2")

    model3 = build_model(X_train, "Adam", "mse", "tanh", 1)
    history = model3.fit(X_train, Y_train3, batch_size=32, epochs=1500, verbose=2)
    plotter(history, "model3")

    # model evaluation
    # NN 1 model results
    predicted = model_results(model1, X_test, Y_test1)
    print("apg443 R2 score: ", r2_score(Y_test1[:, [0]], predicted[:, [0]]) * 100)
    print("bbp443 R2 score: ", r2_score(Y_test1[:, [1]], predicted[:, [1]]) * 100)
    prediction_plotter(Y_test1[:, [0]], predicted[:, [0]], "model1", "apg443")
    prediction_plotter(Y_test1[:, [1]], predicted[:, [1]], "model1", "bbp443")

    # NN 2 model results
    predicted = model_results(model2, X_test, Y_test2)
    print("R(aph/adg) R2 Score:", r2_score(Y_test2, predicted) * 100)
    prediction_plotter(Y_test2, predicted, "model2", "R(aph/adg)")

    # NN 3 model results
    predicted = model_results(model3, X_test, Y_test3)
    print("R(ad/ag) R2 Score:", r2_score(Y_test3, predicted) * 100)
    prediction_plotter(Y_test3, predicted, "model3", "R(ad/ag)")

    # final IOP properties
    org_aph443 = 10 ** test["aph443"].to_numpy()
    org_adg443 = 10 ** test["adg443"].to_numpy()
    org_apg443 = 10 ** test["apg443"].to_numpy()
    org_bbp443 = 10 ** test["bbp443"].to_numpy()
    org_ad443 = 10 ** test["ad443"].to_numpy()
    org_ag443 = 10 ** test["ag443"].to_numpy()

    # NN1 - Level 1 products
    Tot = Y_train1_scaler.inverse_transform(model1.predict(X_test))
    pred_apg443 = 10 ** Tot[:, 0]
    pred_bbp443 = 10 ** Tot[:, 1]
    iop_product_statistics(
        org_apg443,
        org_bbp443,
        pred_apg443,
        pred_bbp443,
        "model1_level1",
        "apg443",
        "bbp443",
        [2.68, 2.50, 2.32, 2.14, 1.96],
        [0.115, 0.107, 0.099, 0.091, 0.083],
    )
    # NN2 - Level 2 products
    ratio_aph443_adg443 = (
        10 ** Y_train2_scaler.inverse_transform(model2.predict(X_test))[:, 0]
    )
    pred_aph443 = np.divide(pred_apg443, (1.0 + (1.0 / ratio_aph443_adg443)))
    pred_adg443 = np.subtract(pred_apg443, pred_aph443)
    iop_product_statistics(
        org_aph443,
        org_adg443,
        pred_aph443,
        pred_adg443,
        "model2_level2",
        "aph443",
        "adg443",
        [0.45, 0.42, 0.39, 0.36, 0.33],
        [2.3, 2.15, 2.0, 1.85, 1.7],
    )
    # NN3 - Level 3 products
    ratio_ad443_ag443 = (
        10 ** Y_train3_scaler.inverse_transform(model3.predict(X_test))[:, 0]
    )
    pred_ad443 = np.divide(pred_adg443, (1.0 + (1.0 / ratio_ad443_ag443)))
    pred_ag443 = np.subtract(pred_adg443, pred_ad443)
    iop_product_statistics(
        org_ad443,
        org_ag443,
        pred_ad443,
        pred_ag443,
        "model3_level3",
        "ad443",
        "ag443",
        [0.40, 0.37, 0.34, 0.31, 0.28],
        [2.2, 2.05, 1.9, 1.75, 1.6],
    )

    # save the model to disk
    model1.save("./model/model1")
    model2.save("./model/model2")
    model3.save("./model/model3")

    # load the model from disk
    model1 = tf.keras.models.load_model("./model/model1")
    model2 = tf.keras.models.load_model("./model/model2")
    model3 = tf.keras.models.load_model("./model/model3")


if __name__ == "__main__":
    main()
