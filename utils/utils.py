# All necessary imports
import numpy as np
import pandas as pd
import os
import re
import types
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from time import perf_counter
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.colors as mcolors
import random
import pickle
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras import Model as KerasModel
from scipy.interpolate import interp1d
import warnings
from ucimlrepo import fetch_ucirepo
from collections import defaultdict
from scipy.stats import qmc
import gc
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# This function is used to plot the decision boundaries of the models trained for different values of alpha.
# It appears in the two-dimensional datasets and called during the computations of Experiment 2.
def plot_decision_boundary_conti(model, ax, xlim=(-1, 1), ylim=(-1, 1), h=0.01):
    """
    Plot the decision boundaries of the models trained for different values of alpha.

    Arguments:
        model: FunctionType or tensorflow.keras.Model or others --- Model to plot its decision boundary.
        ax: array of matplotlib.axes.Axes --- Axes of the subfigure in a larger plot.
        xlim: 2-tuple --- Limits of the x-axis that determine where to plot the boundary.
        ylim: 2-tuple --- Limits of the y-axis that determine where to plot the boundary.
        h: float --- Granularity of the plot.

    Returns:
        Nothing.
    """
    xx, yy = np.meshgrid(np.arange(xlim[0], xlim[1], h),
                         np.arange(ylim[0], ylim[1], h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict scores
    if isinstance(model, types.FunctionType):
        Z = model(grid_points)
    elif isinstance(model, tf.keras.Model):
        Z = model.predict(grid_points, verbose=0)
    else:
        Z = model.predict(grid_points)

    Z = np.ravel(Z).astype(float)

    # Binarize sigmoid output if necessary
    if np.all((Z >= 0) & (Z <= 1)):
        binary_Z = (Z > 0.5).astype(float)
        Z = 2 * binary_Z - 1  # Map to {-1, 1}

    Z = Z.reshape(xx.shape)

    max_abs = np.max(np.abs(Z))
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    ax.contourf(xx, yy, Z, levels=50, cmap='coolwarm', norm=norm, alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# The function below loads the metrics stored for the different values of alpha and averages them.
def load_and_average_results(folder_path, file_prefix):
    """
    Load the metrics stored for the different values of alpha and average them.

    Arguments:
        folder_path: str --- Path to the folder where results are stored.
        file_prefix: str --- Name of the results file truncated to remove the seed number. 

    Returns:
        acc_avg: numpy.ndarray --- Array of averaged accuracies. 
        efe_unif_avg: numpy.ndarray --- Array of averaged empirical fidelity errors on a uniform dataset.
    """
    
    acc_all = []
    efe_unif_all = []

    files = [f for f in os.listdir(folder_path) if f.startswith(file_prefix) and f.endswith(".pkl")]
    if not files:
        raise FileNotFoundError(f"No result files found with prefix '{file_prefix}' in '{folder_path}'")

    for fname in files:
        with open(os.path.join(folder_path, fname), "rb") as f:
            data = pickle.load(f)
            acc_all.append(np.array(data["acc"]))
            efe_unif_all.append(np.array(data["efe_unif"]))

    acc_all = np.array(acc_all)
    efe_unif_all = np.array(efe_unif_all)

    acc_avg = np.mean(acc_all, axis=0)
    efe_unif_avg = np.mean(efe_unif_all, axis=0)

    return acc_avg, efe_unif_avg

# The function below can take the averaged metrics for different values of alpha provided by the previous one. 
# Its objective is to print them as a table and to generate plots. 
def plot_avg_acc_vs_efe_unif(acc_avg, efe_unif_avg):
    """
    Print tables and generate line plots for the given metrics.

    Arguments:
        acc_avg: numpy.ndarray --- Array of averaged accuracies. 
        efe_unif_avg: numpy.ndarray --- Array of averaged empirical fidelity errors on a uniform dataset.

    Returns:
        Nothing.
    """
    
    x_vals = ["BB", "HC"] + [f"{a}" for a in [0, 0.25, 0.5, 0.75, 1, 1.25]]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_acc = 'tab:blue'
    color_efeu = 'tab:red'

    ax1.set_xlabel("Model")
    ax1.set_ylabel("Accuracy", color=color_acc)
    ax1.plot(x_vals, acc_avg, marker='s', color=color_acc, label="Accuracy")
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.set_ylabel("1-EFE Unif", color=color_efeu)
    ax2.plot(x_vals, 1-efe_unif_avg, marker='^', color=color_efeu, label="1-EFE Unif")
    ax2.tick_params(axis='y', labelcolor=color_efeu)
    ax2.set_yscale("log")

    fig.suptitle("Average Accuracy and 1-EFE on Uniform Samples (log scale)")
    fig.tight_layout()
    plt.show()

    df = pd.DataFrame({
        "Model": x_vals,
        "Accuracy (avg)": acc_avg,
        "EFE Unif (avg)": efe_unif_avg,
        "1-EFE Unif": 1-efe_unif_avg
    })

    print("\n=== Average Accuracy and EFE Uniform Table ===")
    print(df.to_string(index=False))


# This function trains several models continuously across different values of alpha with the same two-dimensional dataset.
# It does so for Large Neural Network copies and plots the resulting decision boundaries.
def train_copy_LNN_conti(data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train LNN models in a two-dimensional dataset across preselected values of alpha and plots them. 

    Arguments:
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns: 
        acc: list --- List of accuracies achieved by the LNN models for different values of alpha.
        efe_unif: list --- List of empirical fidelity errors on a uniform dataset achieved by the LNN models for different values of alpha.
    """
    acc = []
    efe_unif = []

    alphas = [0, 0.25, 0.5, 0.75, 1, 1.25]

    # Setup for plotting decision boundaries
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Decision Boundaries', fontsize=16)
    axs = axs.flatten()

    # --- Black box model ---
    plot_decision_boundary_conti(bbmodelW, axs[0])
    axs[0].set_title("Black box")

    pred_bb = bbmodelW(X_test)
    pred_bb_syn = bbmodelW(data_test_syn)
    acc.append(np.mean(y_test == ((pred_bb + 1) // 2)))  # Convert {-1,1} to {0,1}
    efe_unif.append(np.mean(y_test_syn != pred_bb_syn))

    # --- Hard copy model ---
    hard_copy_model = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
 
    hard_copy_model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy())
    y_binary = np.where(lab > 0, 1, 0)
    hard_copy_model.fit(data, y_binary, batch_size=32, epochs=5, verbose=0)

    acc.append(np.mean(y_test == np.where(hard_copy_model(X_test) > 0.5, 1, 0).flatten()))
    efe_unif.append(np.mean(y_test_syn != np.where(hard_copy_model(data_test_syn) > 0.5, 1, -1).flatten()))
    print("Computations done for the hard copy")

    plot_decision_boundary_conti(hard_copy_model, axs[1])
    axs[1].set_title("Hard copy")
    del hard_copy_model
    gc.collect()

    # --- Alpha models ---
    for i, alpha in enumerate(alphas):
        model = keras.Sequential(
            [
                layers.Dense(512, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="linear"),
            ]
        )
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        model.fit(data, np.sign(lab) * np.abs(lab) ** alpha, batch_size=32, epochs=5, verbose=0)

        acc.append(np.mean((2 * y_test - 1) == (np.sign(model(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model(data_test_syn)).flatten())))
        print(f"Computations done for the α = {alpha} copy")

        plot_decision_boundary_conti(model, axs[i + 2])
        axs[i + 2].set_title(f"α = {alpha}")
        del model
        gc.collect()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return acc, efe_unif

# This function trains several models continuously across different values of alpha with the same two-dimensional dataset.
# It does so for Medium Neural Network copies and plots the resulting decision boundaries.
def train_copy_MNN_conti(data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train MNN models in a two-dimensional dataset across preselected values of alpha and plots them.
    
    Arguments:
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 
        
    Returns: 
        acc: list --- List of accuracies achieved by the MNN models for different values of alpha.
        efe_unif: list --- List of empirical fidelity errors on a uniform dataset achieved by the MNN models for different values of alpha.
    """
    acc = []
    efe_unif = []

    alphas = [0, 0.25, 0.5, 0.75, 1, 1.25]

    # Setup for plotting decision boundaries
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Decision Boundaries', fontsize=16)
    axs = axs.flatten()

    # --- Black box model ---
    plot_decision_boundary_conti(bbmodelW, axs[0])
    axs[0].set_title("Black box")

    pred_bb = bbmodelW(X_test)
    pred_bb_syn = bbmodelW(data_test_syn)
    acc.append(np.mean(y_test == ((pred_bb + 1) // 2)))  # Convert {-1,1} to {0,1}
    efe_unif.append(np.mean(y_test_syn != pred_bb_syn))

    # --- Hard copy model ---
    hard_copy_model = keras.Sequential(
        [
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
 
    hard_copy_model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy())
    y_binary = np.where(lab > 0, 1, 0)
    hard_copy_model.fit(data, y_binary, batch_size=32, epochs=5, verbose=0)

    acc.append(np.mean(y_test == np.where(hard_copy_model(X_test) > 0.5, 1, 0).flatten()))
    efe_unif.append(np.mean(y_test_syn != np.where(hard_copy_model(data_test_syn) > 0.5, 1, -1).flatten()))
    print("Computations done for the hard copy")

    plot_decision_boundary_conti(hard_copy_model, axs[1])
    axs[1].set_title("Hard copy")
    del hard_copy_model
    gc.collect()

    # --- Alpha models ---
    for i, alpha in enumerate(alphas):
        model = keras.Sequential(
            [
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="linear"),
            ]
        )
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        model.fit(data, np.sign(lab) * np.abs(lab) ** alpha, batch_size=32, epochs=5, verbose=0)

        acc.append(np.mean((2 * y_test - 1) == (np.sign(model(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model(data_test_syn)).flatten())))
        print(f"Computations done for the α = {alpha} copy")

        plot_decision_boundary_conti(model, axs[i + 2])
        axs[i + 2].set_title(f"α = {alpha}")
        del model
        gc.collect()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return acc, efe_unif


# This function trains several models continuously across different values of alpha with the same two-dimensional dataset.
# It does so for Small Neural Network copies and plots the resulting decision boundaries.
def train_copy_SNN_conti(data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train SNN models in a two-dimensional dataset across preselected values of alpha and plots them.
    
    Arguments:
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns: 
        acc: list --- List of accuracies achieved by the SNN models for different values of alpha.
        efe_unif: list --- List of empirical fidelity errors on a uniform dataset achieved by the SNN models for different values of alpha.
    """
    acc = []
    efe_unif = []

    alphas = [0, 0.25, 0.5, 0.75, 1, 1.25]

    # Setup for plotting decision boundaries
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Decision Boundaries', fontsize=16)
    axs = axs.flatten()

    # --- Black box model ---
    plot_decision_boundary_conti(bbmodelW, axs[0])
    axs[0].set_title("Black box")

    pred_bb = bbmodelW(X_test)
    pred_bb_syn = bbmodelW(data_test_syn)
    acc.append(np.mean(y_test == ((pred_bb + 1) // 2)))  # Convert {-1,1} to {0,1}
    efe_unif.append(np.mean(y_test_syn != pred_bb_syn))

    # --- Hard copy model ---
    hard_copy_model = keras.Sequential(
        [
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
 
    hard_copy_model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy())
    y_binary = np.where(lab > 0, 1, 0)
    hard_copy_model.fit(data, y_binary, batch_size=32, epochs=5, verbose=0)

    acc.append(np.mean(y_test == np.where(hard_copy_model(X_test) > 0.5, 1, 0).flatten()))
    efe_unif.append(np.mean(y_test_syn != np.where(hard_copy_model(data_test_syn) > 0.5, 1, -1).flatten()))
    print("Computations done for the hard copy")

    plot_decision_boundary_conti(hard_copy_model, axs[1])
    axs[1].set_title("Hard copy")
    del hard_copy_model
    gc.collect()

    # --- Alpha models ---
    for i, alpha in enumerate(alphas):
        model = keras.Sequential(
            [
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="linear"),
            ]
        )
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        model.fit(data, np.sign(lab) * np.abs(lab) ** alpha, batch_size=32, epochs=5, verbose=0)

        acc.append(np.mean((2 * y_test - 1) == (np.sign(model(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model(data_test_syn)).flatten())))
        print(f"Computations done for the α = {alpha} copy")

        plot_decision_boundary_conti(model, axs[i + 2])
        axs[i + 2].set_title(f"α = {alpha}")
        del model
        gc.collect()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return acc, efe_unif

# This function trains several models continuously across different values of alpha with the same two-dimensional dataset.
# It does so for Gradient Boosting copies and plots the resulting decision boundaries.
def train_copy_GB_conti(data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train GB models in a two-dimensional dataset across preselected values of alpha and plots them.
    
    Arguments:
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns: 
        acc: list --- List of accuracies achieved by the GB models for different values of alpha.
        efe_unif: list --- List of empirical fidelity errors on a uniform dataset achieved by the GB models for different values of alpha.
    """
    acc = []
    efe_unif = []

    alphas = [0, 0.25, 0.5, 0.75, 1, 1.25]

    # Setup for plotting decision boundaries
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Decision Boundaries', fontsize=16)
    axs = axs.flatten()

    # --- Black box model ---
    plot_decision_boundary_conti(bbmodelW, axs[0])
    axs[0].set_title("Black box")

    pred_bb = bbmodelW(X_test)
    pred_bb_syn = bbmodelW(data_test_syn)
    acc.append(np.mean(y_test == ((pred_bb + 1) // 2)))  # Convert {-1,1} to {0,1}
    efe_unif.append(np.mean(y_test_syn != pred_bb_syn))

    # --- Hard copy model ---
    hard_copy_model = HistGradientBoostingClassifier()
    y_binary = np.where(lab > 0, 1, 0)
    hard_copy_model.fit(data, y_binary)

    acc.append(np.mean(y_test == np.where(hard_copy_model.predict(X_test) > 0.5, 1, 0).flatten()))
    efe_unif.append(np.mean(y_test_syn != np.where(hard_copy_model.predict(data_test_syn) > 0.5, 1, -1).flatten()))
    print("Computations done for the hard copy")
 
    plot_decision_boundary_conti(hard_copy_model, axs[1])
    axs[1].set_title("Hard copy")
    del hard_copy_model
    gc.collect()

    # --- Alpha models ---
    for i, alpha in enumerate(alphas):
        model = HistGradientBoostingRegressor()
        model.fit(data, np.sign(lab) * np.abs(lab) ** alpha)

        acc.append(np.mean((2 * y_test - 1) == (np.sign(model.predict(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model.predict(data_test_syn)).flatten())))
        print(f"Computations done for the α = {alpha} copy")

        plot_decision_boundary_conti(model, axs[i + 2])
        axs[i + 2].set_title(f"α = {alpha}")
        del model
        gc.collect()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return acc, efe_unif

# This function trains several models continuously across different values of alpha with the same high-dimensional dataset.
# It does so for Large Neural Network copies. Contrary to the previous functions, this one produces no plots.
def train_copy_LNN_conti_hdim(data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train LNN models across preselected values of alpha.
    
    Arguments:
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns: 
        acc: list --- List of accuracies achieved by the LNN models for different values of alpha.
        efe_unif: list --- List of empirical fidelity errors on a uniform dataset achieved by the LNN models for different values of alpha.
    """
    acc = []
    efe_unif = []

    alphas = [0, 0.25, 0.5, 0.75, 1, 1.25]

    # --- Black box model ---

    pred_bb = bbmodelW(X_test)
    pred_bb_syn = bbmodelW(data_test_syn)
    acc.append(np.mean(y_test == ((pred_bb + 1) // 2)))  # Convert {-1,1} to {0,1}
    efe_unif.append(np.mean(y_test_syn != pred_bb_syn))

    # --- Hard copy model ---
    hard_copy_model = keras.Sequential(
        [
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
 
    hard_copy_model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy())
    y_binary = np.where(lab > 0, 1, 0)
    hard_copy_model.fit(data, y_binary, batch_size=32, epochs=5, verbose=0)

    acc.append(np.mean(y_test == np.where(hard_copy_model(X_test) > 0.5, 1, 0).flatten()))
    efe_unif.append(np.mean(y_test_syn != np.where(hard_copy_model(data_test_syn) > 0.5, 1, -1).flatten()))
    print("Computations done for the hard copy")

    del hard_copy_model
    gc.collect()

    # --- Alpha models ---
    for i, alpha in enumerate(alphas):
        model = keras.Sequential(
            [
                layers.Dense(512, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="linear"),
            ]
        )
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        model.fit(data, np.sign(lab) * np.abs(lab) ** alpha, batch_size=32, epochs=5, verbose=0)

        acc.append(np.mean((2 * y_test - 1) == (np.sign(model(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model(data_test_syn)).flatten())))
        print(f"Computations done for the α = {alpha} copy")

        del model
        gc.collect()

    return acc, efe_unif


# This function trains several models continuously across different values of alpha with the same high-dimensional dataset.
# It does so for Medium Neural Network copies. Contrary to the previous functions, this one produces no plots.
def train_copy_MNN_conti_hdim(data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train MNN models across preselected values of alpha.
    
    Arguments:
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns: 
        acc: list --- List of accuracies achieved by the MNN models for different values of alpha.
        efe_unif: list --- List of empirical fidelity errors on a uniform dataset achieved by the MNN models for different values of alpha.
    """
    acc = []
    efe_unif = []

    alphas = [0, 0.25, 0.5, 0.75, 1, 1.25]

    # --- Black box model ---
    pred_bb = bbmodelW(X_test)
    pred_bb_syn = bbmodelW(data_test_syn)
    acc.append(np.mean(y_test == ((pred_bb + 1) // 2)))  # Convert {-1,1} to {0,1}
    efe_unif.append(np.mean(y_test_syn != pred_bb_syn))

    # --- Hard copy model ---
    hard_copy_model = keras.Sequential(
        [
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
 
    hard_copy_model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy())
    y_binary = np.where(lab > 0, 1, 0)
    hard_copy_model.fit(data, y_binary, batch_size=32, epochs=5, verbose=0)

    acc.append(np.mean(y_test == np.where(hard_copy_model(X_test) > 0.5, 1, 0).flatten()))
    efe_unif.append(np.mean(y_test_syn != np.where(hard_copy_model(data_test_syn) > 0.5, 1, -1).flatten()))
    print("Computations done for the hard copy")

    del hard_copy_model
    gc.collect()

    # --- Alpha models ---
    for i, alpha in enumerate(alphas):
        model = keras.Sequential(
            [
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="linear"),
            ]
        )
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        model.fit(data, np.sign(lab) * np.abs(lab) ** alpha, batch_size=32, epochs=5, verbose=0)

        acc.append(np.mean((2 * y_test - 1) == (np.sign(model(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model(data_test_syn)).flatten())))
        print(f"Computations done for the α = {alpha} copy")
        
        del model
        gc.collect()

    return acc, efe_unif

# This function trains several models continuously across different values of alpha with the same high-dimensional dataset.
# It does so for Small Neural Network copies. Contrary to the previous functions, this one produces no plots.
def train_copy_SNN_conti_hdim(data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train SNN models across preselected values of alpha.
    
    Arguments:
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns: 
        acc: list --- List of accuracies achieved by the SNN models for different values of alpha.
        efe_unif: list --- List of empirical fidelity errors on a uniform dataset achieved by the SNN models for different values of alpha.
    """
    acc = []
    efe_unif = []

    alphas = [0, 0.25, 0.5, 0.75, 1, 1.25]

    # --- Black box model ---
    pred_bb = bbmodelW(X_test)
    pred_bb_syn = bbmodelW(data_test_syn)
    acc.append(np.mean(y_test == ((pred_bb + 1) // 2)))  # Convert {-1,1} to {0,1}
    efe_unif.append(np.mean(y_test_syn != pred_bb_syn))

    # --- Hard copy model ---
    hard_copy_model = keras.Sequential(
        [
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
 
    hard_copy_model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy())
    y_binary = np.where(lab > 0, 1, 0)
    hard_copy_model.fit(data, y_binary, batch_size=32, epochs=5, verbose=0)

    acc.append(np.mean(y_test == np.where(hard_copy_model(X_test) > 0.5, 1, 0).flatten()))
    efe_unif.append(np.mean(y_test_syn != np.where(hard_copy_model(data_test_syn) > 0.5, 1, -1).flatten()))
    print("Computations done for the hard copy")
    
    del hard_copy_model
    gc.collect()

    # --- Alpha models ---
    for i, alpha in enumerate(alphas):
        model = keras.Sequential(
            [
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="linear"),
            ]
        )
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        model.fit(data, np.sign(lab) * np.abs(lab) ** alpha, batch_size=32, epochs=5, verbose=0)

        acc.append(np.mean((2 * y_test - 1) == (np.sign(model(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model(data_test_syn)).flatten())))
        print(f"Computations done for the α = {alpha} copy")

        del model
        gc.collect()

    return acc, efe_unif

# This function trains several models continuously across different values of alpha with the same high-dimensional dataset.
# It does so for Gradient Boosting copies. Contrary to the previous functions, this one produces no plots.
def train_copy_GB_conti_hdim(data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train GB models across preselected values of alpha.
    
    Arguments:
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns: 
        acc: list --- List of accuracies achieved by the GB models for different values of alpha.
        efe_unif: list --- List of empirical fidelity errors on a uniform dataset achieved by the GB models for different values of alpha.
    """
    acc = []
    efe_unif = []

    alphas = [0, 0.25, 0.5, 0.75, 1, 1.25]

    # --- Black box model ---
    pred_bb = bbmodelW(X_test)
    pred_bb_syn = bbmodelW(data_test_syn)
    acc.append(np.mean(y_test == ((pred_bb + 1) // 2)))  # Convert {-1,1} to {0,1}
    efe_unif.append(np.mean(y_test_syn != pred_bb_syn))

    # --- Hard copy model ---
    hard_copy_model = HistGradientBoostingClassifier()
    y_binary = np.where(lab > 0, 1, 0)
    hard_copy_model.fit(data, y_binary)

    acc.append(np.mean(y_test == np.where(hard_copy_model.predict(X_test) > 0.5, 1, 0).flatten()))
    efe_unif.append(np.mean(y_test_syn != np.where(hard_copy_model.predict(data_test_syn) > 0.5, 1, -1).flatten()))
    print("Computations done for the hard copy")

    del hard_copy_model
    gc.collect()

    # --- Alpha models ---
    for i, alpha in enumerate(alphas):
        model = HistGradientBoostingRegressor()
        model.fit(data, np.sign(lab) * np.abs(lab) ** alpha)

        acc.append(np.mean((2 * y_test - 1) == (np.sign(model.predict(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model.predict(data_test_syn)).flatten())))
        print(f"Computations done for the α = {alpha} copy")

        del model
        gc.collect()

    return acc, efe_unif

# This function is used to generate the first synthetic and two-dimensional dataset used in the experiments. 
def generate_dataset_1(n_points):
    """
    Generate two 2D Gaussian clusters that present a certain overlapping.
    
    Arguments:
        n_points: int --- Number that controls the size of the synthetic dataset generated. 

    Returns: 
        X: numpy.ndarray --- Points of the synthetic dataset.
        y: numpy.ndarray --- Targets of the synthetic dataset.
    """
    # Generate two clusters with different means and variances
    cluster1 = np.random.normal(loc=-2, scale=2, size=(n_points, 2))
    cluster2 = np.random.normal(loc=1, scale=1, size=(n_points, 2))

    # Combine them to build the dataset
    X = np.vstack((cluster1, cluster2))
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))

    # Normalize the dataset
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    X = 2 * (X - min_vals) / (max_vals - min_vals) - 1 

    # Shuffle the data
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    return X, y

# This function is used to generate the second synthetic and two-dimensional dataset used in the experiments. 
def generate_dataset_2(n_points, noise=0.5):
    """
    Generate a two spirals shaped dataset.
        
    Arguments:
        n_points: int --- Number that controls the size of the synthetic dataset generated. 
        noise: float --- Parameter that determines the width of the spirals.

    Returns: 
        data: numpy.ndarray --- Points of the synthetic dataset.
        labels: numpy.ndarray --- Targets of the synthetic dataset.
    """
    # Generate spiral points
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y =  np.sin(n) * n + np.random.rand(n_points, 1) * noise

    # Combine the above points to build the dataset
    data = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    labels = np.hstack((np.zeros(n_points), np.ones(n_points)))

    # Normalize the dataset
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    data = 2 * (data - min_vals) / (max_vals - min_vals) - 1

    return data, labels

# This function is used to generate the third synthetic and two-dimensional dataset used in the experiments. 
def generate_dataset_3(n_points, resolution=200):
    """
    Generate a space-filling and convoluted dataset, with a complex but smooth decision boundary.
        
    Arguments:
        n_points: int --- Number that controls the size of the synthetic dataset generated.
        resolution: int --- Parameter that controls the complexity of the dataset generated.

    Returns: 
        X: numpy.ndarray --- Points of the synthetic dataset.
        y: numpy.ndarray --- Targets of the synthetic dataset.
    """
    # Create a random matrix
    field = np.random.rand(resolution, resolution)

    # Smooth it using Gaussian filters
    smooth_field = gaussian_filter(field, sigma=5)

    # Normalize the result to ensure the entries are in [0,1]
    smooth_field = (smooth_field - smooth_field.min()) / (smooth_field.max() - smooth_field.min())

    # Generate the random points of the dataset
    X = np.random.uniform(-1, 1, (n_points, 2))

    # Map them to the corresponding indices in the matrix
    grid_x = ((X[:, 0] + 1) / 2 * (resolution - 1)).astype(int)
    grid_y = ((X[:, 1] + 1) / 2 * (resolution - 1)).astype(int)

    # Use the matrix values at these indices to determine the labels
    values = smooth_field[grid_y, grid_x]
    y = (values > 0.5).astype(int)

    return X, y

# This function loads the required results saved by Experiment 1 notebooks.
# The output of this function is used to make tables and plots with this results.
def load_all_seeds(folder_path, file_prefix):
    """
    Load all seed files from the folder with the given prefix.

    Arguments:
        folder_path: str --- Path to the folder where results are stored.
        file_prefix: str --- Name of the results file truncated to remove the seed number. 
    
    Returns:
        model_data: dict --- Dictionary with the loaded model metrics
        blackb_acc: list --- List of accuracy values of the black boxes
    """
    seed_files = [f for f in os.listdir(folder_path) if f.startswith(file_prefix)]
    if not seed_files:
        raise FileNotFoundError(f"No files starting with '{file_prefix}' found in {folder_path}")

    model_data = {
        model_id: {
            "pts": [],
            "efe_unif": [],
            "efe": [],
            "acc": []
        } for model_id in [1, 2, 3]
    }

    blackb_acc = []

    for fname in seed_files:
        with open(os.path.join(folder_path, fname), "rb") as f:
            data_loaded = pickle.load(f)
        for model_id in [1, 2, 3]:
            m = data_loaded.get(f"model{model_id}")
            if m is None:
                continue
            model_data[model_id]["pts"].append(np.array(m["pts"]))
            model_data[model_id]["efe_unif"].append(np.array(m["efe_unif"]))
            model_data[model_id]["efe"].append(np.array(m["efe"]))
            model_data[model_id]["acc"].append(np.array(m["acc"]))

        blackb = data_loaded.get("blackb")
        if blackb is not None and "acc" in blackb:
            blackb_acc.append(float(blackb["acc"]))

    return model_data, blackb_acc

# This function creates the line plots shown as part of the results of Experiment 1.
# The line plots are built with the results of the five different runs considered.
# These results are interpolated and the plots only show values where all five runs could be averaged. 
def average_metric_interpolated(x_seeds, y_seeds, num_points=100, log_scale=True):
    """
    Build line plots interpolating and averaging the given results.

    Arguments: 
        x_seeds: list --- List of arrays that contain the training dataset sizes used for each seed.
        y_seeds: list --- List of arrays that contain the achieved metrics at each size for each seed.
        num_points: int --- Number of points used in the interpolation.
        log_scale: bool --- Bool that determines if logarithmic scale should be used or not during the interpolation.

    Returns:
        x_common: numpy.ndarray --- Sequence of values that constitutes the x axis of the plot.
        y_avg: numpy.ndarray --- Interpolated and averaged results that form the y axis of the plot. 
    """
    
    x_mins = [x[0] for x in x_seeds if len(x) > 0]
    x_maxs = [x[-1] for x in x_seeds if len(x) > 0]
    if not x_mins or not x_maxs:
        return None, None

    x_min = max(x_mins)
    x_max = min(x_maxs)
    if x_max <= x_min:
        return None, None

    if log_scale:
        x_common = np.logspace(np.log10(x_min), np.log10(x_max), num_points)
    else:
        x_common = np.linspace(x_min, x_max, num_points)

    y_interp_all = []
    for x, y in zip(x_seeds, y_seeds):
        if len(x) < 2:
            continue
        f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
        y_interp = f(x_common)
        y_interp_all.append(y_interp)

    y_interp_all = np.array(y_interp_all)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        y_avg = np.nanmean(y_interp_all, axis=0)

    return x_common, y_avg

# This function computes the means and standard deviations of the results it receives as arguments.
# The output is returned in a table form that can be easily printed.
def compute_final_stats(model_data, blackb_acc):
    """
    Compute mean and std dev of final metric values across seeds.

    Arguments:
        model_data: dict --- Dictionary with the loaded model metrics
        blackb_acc: list --- List of accuracy values of the black boxes
    
    Returns:
        df: pandas.DataFrame --- A dataframe with the averaged metrics for each model.
    """
    metric_list = ["acc", "efe", "efe_unif", "pts"]
    column_order = ["model"]
    for metric in metric_list:
        column_order.append(f"{metric} m.")
        column_order.append(f"{metric} std.")

    table_rows = []

    for model_id in [1, 2, 3]:
        row = {"model": f"Model {model_id}"}
        for metric in metric_list:
            values = [arr[-1] for arr in model_data[model_id][metric] if len(arr) > 0]
            if not values:
                mean_val, std_val = "", ""
            else:
                mean_val = f"{np.mean(values):.4f}"
                std_val = f"{np.std(values):.4f}"

            row[f"{metric} m."] = mean_val
            row[f"{metric} std."] = std_val
        table_rows.append(row)

    # Blackbox row (only acc fields)
    if blackb_acc:
        mean_bb = f"{np.mean(blackb_acc):.4f}"
        std_bb = f"{np.std(blackb_acc):.4f}"
    else:
        mean_bb, std_bb = "", ""

    row_bb = {
        "model": "Blackbox",
        "acc m.": mean_bb,
        "acc std.": std_bb,
        "efe m.": "", "efe std.": "",
        "efe_unif m.": "", "efe_unif std.": "",
        "pts m.": "", "pts std.": ""
    }
    table_rows.append(row_bb)

    df = pd.DataFrame(table_rows)[column_order]
    return df

# This is the function that we call to show the results from Experiment 1.
# It combines several of the above functions to load these results, to plot them and to average them.
def plot_all_results(folder_path, file_prefix):
    """
    Load and plot average results for Algorithm 1, Algorithm 2 and Hard copies.
    
    Arguments:
        folder_path: str --- Path to the folder where results are stored.
        file_prefix: str --- Name of the results file truncated to remove the seed number. 
    
    Returns:
        df: pandas.DataFrame --- A dataframe with the averaged metrics for each model.
    """
    model_data, blackb_acc = load_all_seeds(folder_path, file_prefix)

    metrics = ["efe_unif", "efe", "acc"]
    model_ids = [1, 2, 3]
    colors = ['red', 'blue', 'black']
    labels = {
        1: "Algorithm 1 Copy",
        2: "Algorithm 2 Copy",
        3: "Hard Copy"
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for col, metric in enumerate(metrics):
        ax = axs[col]
        for i, model_id in enumerate(model_ids):
            data = model_data[model_id]

            x_pts, y_pts = average_metric_interpolated(
                data["pts"], data[metric], log_scale=True
            )
            if x_pts is not None:
                ax.plot(x_pts, y_pts, label=labels[model_id], color=colors[i])

        ax.set_xlabel("Points")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs Points")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Return final stats table (no bolding)
    summary_df = compute_final_stats(model_data, blackb_acc)
    return summary_df

# This function is a variant of the previous one prepared to be used in the Two-Stage Distance Copying Extension.
# The main differences are visual, where the colors and labels of the plots are adapted to the new setting. 
def plot_all_results_ext(folder_path, file_prefix):
    """
    Load and plot average results for Stage 1, Stage 2 and Hard copies.

    Arguments:
        folder_path: str --- Path to the folder where results are stored.
        file_prefix: str --- Name of the results file truncated to remove the seed number. 
    
    Returns:
        df: pandas.DataFrame --- A dataframe with the averaged metrics for each model.
    """
    
    model_data, blackb_acc = load_all_seeds(folder_path, file_prefix)

    metrics = ["efe_unif", "efe", "acc"]
    model_ids = [1, 2, 3]

    # Updated colors
    colors = ['blue', 'green', 'black']  # New: green for Stage 1, orange for Stage 2

    # Updated labels
    labels = {
        1: "Stage 1 Copy",
        2: "Stage 2 Copy",
        3: "Hard Copy"
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for col, metric in enumerate(metrics):
        ax = axs[col]
        for i, model_id in enumerate(model_ids):
            data = model_data[model_id]

            x_pts, y_pts = average_metric_interpolated(
                data["pts"], data[metric], log_scale=True
            )
            if x_pts is not None:
                ax.plot(x_pts, y_pts, label=labels[model_id], color=colors[i])

        ax.set_xlabel("Points")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs Points")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

    summary_df = compute_final_stats(model_data, blackb_acc)
    return summary_df

# This function is used to plot the decision boundary of a given model.
# I works for all the two-dimensional models considered in this project, including hard and distance-based copies.
# It is mainly used during the exhibition of Experiment 1 results.
def plot_decision_boundary(model, xlim=(-1, 1), ylim=(-1, 1), h=0.01):
    """
    Plot the decision boundary of a model in 2D input space.

    Arguments:
        model: FunctionType or tensorflow.keras.Model or others --- Model to plot its decision boundary.
        xlim: 2-tuple --- Limits of the x-axis that determine where to plot the boundary.
        ylim: 2-tuple --- Limits of the y-axis that determine where to plot the boundary.
        h: float --- Granularity of the plot.

    Returns:
        Nothing.
    """

    # Create meshgrid
    xx, yy = np.meshgrid(np.arange(xlim[0], xlim[1], h),
                         np.arange(ylim[0], ylim[1], h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict scores
    if isinstance(model, types.FunctionType):
        Z = model(grid_points)
    elif isinstance(model, tf.keras.Model):
        Z = model.predict(grid_points, verbose=0)
    else:
        # For other models (e.g. sklearn), call predict without verbose
        Z = model.predict(grid_points)
        
    # Ensure Z is 1D or flatten if needed
    Z = np.ravel(Z).astype(float)

    if np.all((Z >= 0) & (Z <= 1)):
        binary_Z = (Z > 0.5).astype(float)
        Z = 2 * binary_Z - 1

    Z = Z.reshape(xx.shape)

    # Set color normalization to center colormap at zero
    max_abs = np.max(np.abs(Z))
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    # Plot decision regions
    plt.contourf(xx, yy, Z, levels=50, cmap='coolwarm', norm=norm, alpha=0.5)

    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)

    # Formatting
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gcf().set_size_inches(6, 6)
    plt.show()

# This function generates points around the origin in the unit ball and also returns the distances of the points to the origin.
def generate(ndim, npts):
    """
    Generate a cloud of points in the unit ball.
    
    Arguments:
        ndim: int --- Dimensionality of the cloud.
        npts: int --- Number of points that the generated cloud should have.
    
    Returns:
        pts: numpy.ndarray --- Points that constitute the cloud.
        uni: numpy.ndarray --- Distances to the origin for the points in the cloud. 
    """
    
    vec= np.random.randn(ndim,npts)
    uni = np.random.uniform(0,1,(npts))
    norm = np.linalg.norm(vec, axis=0)
    norm = norm/uni
    pts = vec/norm
    return pts, uni

# This function is used in the implementation of Algorithm 1.
# It looks up to distance d from the previous point and finds the closest point to it with different label. 
# Then, iteratively, centers the cloud of points around the newly found point and repeats the process. 
# It outputs an approximation to the distance between pt and the decision boundary of the model.
def cdistance(d, pt, rdc, uni, func, itermax):
    """
    Compute the distance from pt to the decision boundary, using Algorithm 1 techniques.

    Arguments:
        d: float --- Maximum distance to which distances to the boundary will be computed.
        pt: numpy.ndarray --- Point whose distance to the decision boundary we aim to compute.
        rdc: numpy.ndarray --- Points that constitute the cloud.
        uni: numpy.ndarray --- Distances to the origin for the points in the cloud. 
        func: FunctionType --- Black box model that determines the decision boundary.
        itermax: int --- Number of iterations that the algorithm should perform.

    Returns:
        result: float --- Computed distance to the decision boundary.
    """
    aux = np.zeros(rdc.shape)
    pt2 = np.zeros(len(rdc[0]))
    
    for i in range(len(rdc[0])):
        aux[:,i] = d*rdc[:,i] + pt[i]
    idx = np.argwhere((abs(func(aux)- func(pt.reshape((1,len(rdc[0])))))).flatten()>0.5)
    if len(idx) == 0:
        return d
    idp = np.argmin(uni[idx])
    pt2[:] = aux[idx[idp]][:]
    
    for ite in range(itermax):
        for i in range(len(rdc[0])):
            aux[:,i] = rdc[:,i] + pt2[i]
        idx = np.argwhere((abs(func(aux)- func(pt2.reshape((1,len(rdc[0])))))).flatten()>0.5)
        if len(idx) == 0:
            return d
        idp = np.argmin(uni[idx])
        pt2[:] = aux[idx[idp]][:]

    result = np.linalg.norm(pt2-pt)
    return result


# This is the implementation of Algorithm 1, that uses the previous function.
# It is prepared to satisfy the needs of Experiment 1, taking snapshots of the results at different dataset sizes. 
# It enforces a limit of 1,000,000 sampled points and 240 seconds.
def generate_distances_algo1(dim, region1, region2, d, itermax, bbmodelW):
    """
    Implementation of Algorithm 1 with dataset size logging and time control (240 seconds).

    Arguments:
        dim: int --- Dimensionality of the synthetic dataset that will be generated.
        region1: float --- Lower limit of the square region of interest.
        region2: float --- Upper limit of the square region of interest.
        d: float --- Maximum distance to which distances to the boundary will be computed.
        itermax: int --- Number of iterations that the algorithm should perform.
        bbmodelW: FunctionType --- Black box model that determines the decision boundary.

    Returns:
        l_pts: list --- List of sizes at which the models should be trained. 
        data2: numpy.ndarray --- Sampled dataset.
        le2: numpy.ndarray --- Targets of the sampled dataset.
    """
    
    l_stops = [50, 200, 500, 1_000, 5_000, 10_000, 50_000, 200_000, 
               400_000, 600_000, 800_000, 1_000_000]
    l_pts = []

    rdc, uni = generate(dim,10_000)
    rdc = rdc.T
    
    n_pow2 = next_power_of_2(1_000_000)
    sampler = qmc.Sobol(d=dim, scramble=False)
    sobol_points = sampler.random_base2(m=int(np.log2(n_pow2)))
    sobol_points = sobol_points[:1_000_000]
    data2 = region1 + (region2 - region1) * sobol_points
    np.random.shuffle(data2)
    
    labels = bbmodelW(data2)
    le2 = []
    
    start = perf_counter()
    for i in range(len(data2)):
        le2.append(cdistance(d, data2[i], rdc, uni, bbmodelW, itermax)*labels[i])
        
        end = perf_counter()
        if i in l_stops:
            l_pts.append(i)
            print("We have labelled", i, "points")

        if (end -start) >= 240:
            l_pts.append(i+1)
            print("We have labelled", i, "points in", round(end-start, 2), "seconds")
            break
            
    le2 = np.array(le2)
    if len(le2) == 1_000_000:
        l_pts.append(1_000_000)

    data2 = data2[:l_pts[-1]]
    le2 = le2[:l_pts[-1]]
    
    indices = np.random.permutation(len(data2))
    data2 = data2[indices]
    le2 = le2[indices]
    
    return l_pts, data2, le2

# This is the implementation of Algorithm 2, that computes the distances to the decision boundary for a clustered dataset.
# It is prepared to satisfy the needs of Experiment 1, taking snapshots of the results at different dataset sizes.
# It enforces a limit of 1,000,000 sampled points and 240 seconds.
# Contrary to the implementation presented in the project, here computations are reorganized to improve performance.
def generate_distances_algo2(dim, region1, region2, n1, n2, n3, par1, par2, bbmodelW, batch_size):
    """
    Implementation of Algorithm 2 with dataset size logging and time control (240 seconds).
    
    Arguments:
        dim: int --- Dimensionality of the synthetic dataset that will be generated.
        region1: float --- Lower limit of the square region of interest.
        region2: float --- Upper limit of the square region of interest.
        n1: int --- Number of cluster to sample.
        n2: int --- Number of points per cluster.
        n3: int --- Number of points of the outer cloud.
        par1: float --- Size of the outer cloud of points.
        par2: float --- Size of the inner cloud of points.
        bbmodelW: FunctionType --- Black box model that determines the decision boundary.
        batch_size: int --- Size of the batch in which computations are divided. 

    Returns:
        l_pts: list --- List of sizes at which the models should be trained. 
        dat2: numpy.ndarray --- Sampled dataset.
        lab: numpy.ndarray --- Targets of the sampled dataset.
    """

    l_stops = [10_000, 50_000, 200_000, 400_000, 600_000, 
    800_000, 1_000_000]
    
    # Logging
    l_pts = [50, 200, 500, 1_000, 5_000]
    
    # Step 1: Generate Sobol base points
    n_pow2 = next_power_of_2(n1)
    sampler = qmc.Sobol(d=dim, scramble=False)
    sobol_points = sampler.random_base2(m=int(np.log2(n_pow2)))
    sobol_points = sobol_points[:n1]
    dat1 = region1 + (region2 - region1) * sobol_points
    np.random.shuffle(dat1)

    # Output arrays
    dat2 = np.empty((n1 * n2, dim))
    lab = np.empty(n1 * n2)

    # Step 2: Generate aux and aux1 (only once, reused with translation)
    r_aux, _ = generate(dim, n3)   # shape: (dim, n3)
    r_aux1, _ = generate(dim, n2)  # shape: (dim, n2)

    aux = (par1 * r_aux.T)   # shape: (n3, dim)
    aux1 = (par2 * r_aux1.T) # shape: (n2, dim)

    # Step 3: Precompute distance matrix between aux1[i] and aux[j]
    dist_matrix = np.linalg.norm(aux1[:, None, :] - aux[None, :, :], axis=2)  # (n2, n3)

    start = perf_counter()
    # Batched version of the loop
    for batch_start in range(0, n1, batch_size):
        batch_end = min(batch_start + batch_size, n1)
        current_batch = dat1[batch_start:batch_end]  # shape: (B, dim)
        B = batch_end - batch_start

        # Translate aux and aux1 for each base point in the batch
        aux_translated_batch = aux[None, :, :] + current_batch[:, None, :]    # shape: (B, n3, dim)
        aux1_translated_batch = aux1[None, :, :] + current_batch[:, None, :]  # shape: (B, n2, dim)

        # Combine all for bbmodelW
        combined_batch = np.concatenate([aux_translated_batch, aux1_translated_batch], axis=1)  # (B, n3+n2, dim)
        combined_batch_flat = combined_batch.reshape(-1, dim)  # (B*(n3+n2), dim)

        labels_flat = bbmodelW(combined_batch_flat)  # shape: (B*(n3+n2),)
        labels = labels_flat.reshape(B, n3 + n2)
        res = labels[:, :n3]       # (B, n3)
        lab_chunk = labels[:, n3:] # (B, n2)

        for b in range(B):
            k = batch_start + b
            base = current_batch[b]
            start_idx = k*n2
            end_idx = (k + 1)*n2

            dat2[start_idx:end_idx] = aux1_translated_batch[b]

            for i in range(n2):
                mismatched = res[b] != lab_chunk[b, i]
                if np.any(mismatched):
                    lab[start_idx + i] = np.min(dist_matrix[i][mismatched])*lab_chunk[b, i]
                else:
                    lab[start_idx + i] = par1*lab_chunk[b, i]

                # Logging
            if end_idx in l_stops:
                l_pts.append(end_idx)
                print("We have labelled", end_idx, "points")

            current_time = perf_counter()
            if (current_time - start) >= 240:
                l_pts.append(end_idx)
                print("We have labelled", end_idx, "points in", round(current_time - start, 2), "seconds")
                dat2 = dat2[:l_pts[-1]]
                lab = lab[:l_pts[-1]]
    
                indices = np.random.permutation(len(dat2))
                dat2 = dat2[indices]
                lab = lab[indices]
                return l_pts, dat2, lab

    dat2 = dat2[:l_pts[-1]]
    lab = lab[:l_pts[-1]]
    
    indices = np.random.permutation(len(dat2))
    dat2 = dat2[indices]
    lab = lab[indices]
    
    return l_pts, dat2, lab


# This function is a variant of the previous Algorithm 2 implementation.
# It allows the computations to run for 600 instead of the previous 240.
# It is used in the Two-Stage Distance Copying Extension.
def generate_distances_algo3(dim, region1, region2, n1, n2, n3, par1, par2, bbmodelW, batch_size):
    """
    Implementation of Algorithm 2 with dataset size logging and time control (600 seconds).
        
    Arguments:
        dim: int --- Dimensionality of the synthetic dataset that will be generated.
        region1: float --- Lower limit of the square region of interest.
        region2: float --- Upper limit of the square region of interest.
        n1: int --- Number of cluster to sample.
        n2: int --- Number of points per cluster.
        n3: int --- Number of points of the outer cloud.
        par1: float --- Size of the outer cloud of points.
        par2: float --- Size of the inner cloud of points.
        bbmodelW: FunctionType --- Black box model that determines the decision boundary.
        batch_size: int --- Size of the batch in which computations are divided. 

    Returns:
        l_pts: list --- List of sizes at which the models should be trained. 
        dat2: numpy.ndarray --- Sampled dataset.
        lab: numpy.ndarray --- Targets of the sampled dataset.
    """

    l_stops = [10_000, 50_000, 200_000, 400_000, 600_000, 
    800_000, 1_000_000]
    
    # Logging
    l_pts = [50, 200, 500, 1_000, 5_000]
    
    # Step 1: Generate Sobol base points
    n_pow2 = next_power_of_2(n1)
    sampler = qmc.Sobol(d=dim, scramble=False)
    sobol_points = sampler.random_base2(m=int(np.log2(n_pow2)))
    sobol_points = sobol_points[:n1]
    dat1 = region1 + (region2 - region1) * sobol_points
    np.random.shuffle(dat1)

    # Output arrays
    dat2 = np.empty((n1 * n2, dim))
    lab = np.empty(n1 * n2)

    # Step 2: Generate aux and aux1 (only once, reused with translation)
    r_aux, _ = generate(dim, n3)   # shape: (dim, n3)
    r_aux1, _ = generate(dim, n2)  # shape: (dim, n2)

    aux = (par1 * r_aux.T)   # shape: (n3, dim)
    aux1 = (par2 * r_aux1.T) # shape: (n2, dim)

    # Step 3: Precompute distance matrix between aux1[i] and aux[j]
    dist_matrix = np.linalg.norm(aux1[:, None, :] - aux[None, :, :], axis=2)  # (n2, n3)

    start = perf_counter()
    # New batched version of the loop
    for batch_start in range(0, n1, batch_size):
        batch_end = min(batch_start + batch_size, n1)
        current_batch = dat1[batch_start:batch_end]  # shape: (B, dim)
        B = batch_end - batch_start

        # Translate aux and aux1 for each base point in the batch
        aux_translated_batch = aux[None, :, :] + current_batch[:, None, :]    # shape: (B, n3, dim)
        aux1_translated_batch = aux1[None, :, :] + current_batch[:, None, :]  # shape: (B, n2, dim)

        # Combine all for bbmodelW
        combined_batch = np.concatenate([aux_translated_batch, aux1_translated_batch], axis=1)  # (B, n3+n2, dim)
        combined_batch_flat = combined_batch.reshape(-1, dim)  # (B*(n3+n2), dim)

        labels_flat = bbmodelW(combined_batch_flat)  # shape: (B*(n3+n2),)
        labels = labels_flat.reshape(B, n3 + n2)
        res = labels[:, :n3]       # (B, n3)
        lab_chunk = labels[:, n3:] # (B, n2)

        for b in range(B):
            k = batch_start + b
            base = current_batch[b]
            start_idx = k*n2
            end_idx = (k + 1)*n2

            dat2[start_idx:end_idx] = aux1_translated_batch[b]

            for i in range(n2):
                mismatched = res[b] != lab_chunk[b, i]
                if np.any(mismatched):
                    lab[start_idx + i] = np.min(dist_matrix[i][mismatched])*lab_chunk[b, i]
                else:
                    lab[start_idx + i] = par1*lab_chunk[b, i]

                # Logging
            if end_idx in l_stops:
                l_pts.append(end_idx)
                print("We have labelled", end_idx, "points")

            current_time = perf_counter()
            if (current_time - start) >= 600:
                l_pts.append(end_idx)
                print("We have labelled", end_idx, "points in", round(current_time - start, 2), "seconds")
                dat2 = dat2[:l_pts[-1]]
                lab = lab[:l_pts[-1]]
    
                indices = np.random.permutation(len(dat2))
                dat2 = dat2[indices]
                lab = lab[indices]
                return l_pts, dat2, lab

    dat2 = dat2[:l_pts[-1]]
    lab = lab[:l_pts[-1]]
    
    indices = np.random.permutation(len(dat2))
    dat2 = dat2[indices]
    lab = lab[indices]
    
    return l_pts, dat2, lab

# This function is used to train the Medium Neural Network distance-based copies in Experiment 1.
# It performs the computations at different training dataset sizes.
def train_copy_MNNd (l_pts, data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train MNN distance-based copies for different training dataset sizes.
        
    Arguments:
        l_pts: list --- List of dataset sizes at which training should be performed.
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns:
        efe: list --- List of computed empirical fidelity errors.
        acc: list --- List of the computed accuracies.
        efe_unif: list --- List of the computed empirical fidelity errors on the uniform dataset. 
        model: tensorflow.keras.Model --- Final model trained.
    """
    
    efe = []
    acc = []
    efe_unif = []
    p = np.log(100/5)/np.log(1/1000)
    a = 100/(1000**p)
    
    for pts in l_pts:
        model = keras.Sequential(
            [
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="linear"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())

        model.fit(data[:pts], lab[:pts], batch_size=32, epochs= int(round(a*pts**p, 2)), verbose=0)
        print("Computations for", pts, "points done")

        # Compute the metrics for the first model.
        efe.append(np.mean(bbmodelW(X_test) != (np.sign(model(X_test)).flatten())))
        acc.append(np.mean((2*y_test-1) == (np.sign(model(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model(data_test_syn)).flatten())))
    
    return efe, acc, efe_unif, model

# This function is used to train the Medium Neural Network hard copies in Experiment 1.
# It performs the computations at different training dataset sizes.
def train_copy_MNNh (data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train MNN hard copies for different training dataset sizes.
            
    Arguments:
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns:
        efe: list --- List of computed empirical fidelity errors.
        acc: list --- List of the computed accuracies.
        efe_unif: list --- List of the computed empirical fidelity errors on the uniform dataset. 
        model: tensorflow.keras.Model --- Final model trained.
    """
    
    efe = []
    acc = []
    efe_unif = []
    lab = (lab + 1)//2
    l_pts = [50, 200, 500, 1_000, 5_000, 10_000, 50_000, 200_000, 400_000, 
             600_000, 800_000, 1_000_000]
    p = np.log(100/5)/np.log(1/1000)
    a = 100/(1000**p)
    
    for pts in l_pts:
        model = keras.Sequential(
            [
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy())

        model.fit(data[:pts], lab[:pts], batch_size=32, epochs= int(round(a*pts**p, 2)), verbose=0)
        print("Computations for", pts, "points done")

        # Compute the metrics for the first model.
        efe.append(np.mean(bbmodelW(X_test) != np.where(model(X_test) > 0.5, 1, -1).flatten()))
        acc.append(np.mean(y_test == np.where(model(X_test) > 0.5, 1, 0).flatten()))
        efe_unif.append(np.mean(y_test_syn != np.where(model(data_test_syn) > 0.5, 1, -1).flatten()))
    
    return efe, acc, efe_unif, model

# This function is used to train the Gradient Boosting distance-based copies in Experiment 1.
# It performs the computations at different training dataset sizes.
def train_copy_GBd (l_pts, data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train GB distance-based copies for different training dataset sizes.
            
    Arguments:
        l_pts: list --- List of dataset sizes at which training should be performed.
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns:
        efe: list --- List of computed empirical fidelity errors.
        acc: list --- List of the computed accuracies.
        efe_unif: list --- List of the computed empirical fidelity errors on the uniform dataset. 
        model: tensorflow.keras.Model --- Final model trained.
    """
    
    efe = []
    acc = []
    efe_unif = []
    
    for pts in l_pts:
        model = HistGradientBoostingRegressor()
        model.fit(data[:pts], lab[:pts])
        print("Computations for", pts, "points done")

        # Compute the metrics for the first model.
        efe.append(np.mean(bbmodelW(X_test) != (np.sign(model.predict(X_test)).flatten())))
        acc.append(np.mean((2*y_test-1) == (np.sign(model.predict(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model.predict(data_test_syn)).flatten())))
    
    return efe, acc, efe_unif, model

# This function is used to train the Gradient Boosting hard copies in Experiment 1.
# It performs the computations at different training dataset sizes.    
def train_copy_GBh (data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train GB hard copies for different training dataset sizes.
            
    Arguments:
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns:
        efe: list --- List of computed empirical fidelity errors.
        acc: list --- List of the computed accuracies.
        efe_unif: list --- List of the computed empirical fidelity errors on the uniform dataset. 
        model: tensorflow.keras.Model --- Final model trained.
    """
    
    efe = []
    acc = []
    efe_unif = []
    l_pts = [50, 200, 500, 1_000, 5_000, 10_000, 50_000, 200_000, 400_000, 
             600_000, 800_000, 1_000_000]
    
    for pts in l_pts:
        model = HistGradientBoostingClassifier()
        model.fit(data[:pts], lab[:pts])
        print("Computations for", pts, "points done")

        # Compute the metrics for the first model.
        efe.append(np.mean(bbmodelW(X_test) != np.where(model.predict(X_test) > 0.5, 1, -1).flatten()))
        acc.append(np.mean(y_test == np.where(model.predict(X_test) > 0.5, 1, 0).flatten()))
        efe_unif.append(np.mean(y_test_syn != np.where(model.predict(data_test_syn) > 0.5, 1, -1).flatten()))
    
    return efe, acc, efe_unif, model


# This function is used to train the Large Neural Network distance-based copies in Experiment 1.
# It performs the computations at different training dataset sizes.
def train_copy_LNNd (l_pts, data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train LNN distance-based copies for different training dataset sizes.
            
    Arguments:
        l_pts: list --- List of dataset sizes at which training should be performed.
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns:
        efe: list --- List of computed empirical fidelity errors.
        acc: list --- List of the computed accuracies.
        efe_unif: list --- List of the computed empirical fidelity errors on the uniform dataset. 
        model: tensorflow.keras.Model --- Final model trained.
    """
    
    efe = []
    acc = []
    efe_unif = []
    p = np.log(100/5)/np.log(1/1000)
    a = 100/(1000**p)
    
    for pts in l_pts:
        model = keras.Sequential(
            [
                layers.Dense(512, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="linear"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())

        model.fit(data[:pts], lab[:pts], batch_size=32, epochs= int(round(a*pts**p, 2)), verbose=0)
        print("Computations for", pts, "points done")

        # Compute the metrics for the first model.
        efe.append(np.mean(bbmodelW(X_test) != (np.sign(model(X_test)).flatten())))
        acc.append(np.mean((2*y_test-1) == (np.sign(model(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model(data_test_syn)).flatten())))
    
    return efe, acc, efe_unif, model

    
# This function is used to train the Large Neural Network hard copies in Experiment 1.
# It performs the computations at different training dataset sizes.
def train_copy_LNNh (data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train LNN hard copies for different training dataset sizes.
            
    Arguments:
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns:
        efe: list --- List of computed empirical fidelity errors.
        acc: list --- List of the computed accuracies.
        efe_unif: list --- List of the computed empirical fidelity errors on the uniform dataset. 
        model: tensorflow.keras.Model --- Final model trained.
    """
    
    efe = []
    acc = []
    efe_unif = []
    lab = (lab + 1)//2
    l_pts = [50, 200, 500, 1_000, 5_000, 10_000, 50_000, 200_000, 400_000, 
             600_000, 800_000, 1_000_000]
    p = np.log(100/5)/np.log(1/1000)
    a = 100/(1000**p)
    
    for pts in l_pts:
        model = keras.Sequential(
            [
                layers.Dense(512, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy())

        model.fit(data[:pts], lab[:pts], batch_size=32, epochs= int(round(a*pts**p, 2)), verbose=0)
        print("Computations for", pts, "points done")

        # Compute the metrics for the first model.
        efe.append(np.mean(bbmodelW(X_test) != np.where(model(X_test) > 0.5, 1, -1).flatten()))
        acc.append(np.mean(y_test == np.where(model(X_test) > 0.5, 1, 0).flatten()))
        efe_unif.append(np.mean(y_test_syn != np.where(model(data_test_syn) > 0.5, 1, -1).flatten()))
    
    return efe, acc, efe_unif, model

# This function is used to train the Small Neural Network distance-based copies in Experiment 1.
# It performs the computations at different training dataset sizes.
def train_copy_SNNd (l_pts, data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train SNN distance-based copies for different training dataset sizes.
            
    Arguments:
        l_pts: list --- List of dataset sizes at which training should be performed.
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns:
        efe: list --- List of computed empirical fidelity errors.
        acc: list --- List of the computed accuracies.
        efe_unif: list --- List of the computed empirical fidelity errors on the uniform dataset. 
        model: tensorflow.keras.Model --- Final model trained.
    """
    
    efe = []
    acc = []
    efe_unif = []
    p = np.log(100/5)/np.log(1/1000)
    a = 100/(1000**p)
    
    for pts in l_pts:
        model = keras.Sequential(
            [
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="linear"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())

        model.fit(data[:pts], lab[:pts], batch_size=32, epochs= int(round(a*pts**p, 2)), verbose=0)
        print("Computations for", pts, "points done")

        # Compute the metrics for the first model.
        efe.append(np.mean(bbmodelW(X_test) != (np.sign(model(X_test)).flatten())))
        acc.append(np.mean((2*y_test-1) == (np.sign(model(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model(data_test_syn)).flatten())))
    
    return efe, acc, efe_unif, model

    
# This function is used to train the Small Neural Network hard copies in Experiment 1.
# It performs the computations at different training dataset sizes.
def train_copy_SNNh (data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train SNN hard copies for different training dataset sizes.
            
    Arguments:
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns:
        efe: list --- List of computed empirical fidelity errors.
        acc: list --- List of the computed accuracies.
        efe_unif: list --- List of the computed empirical fidelity errors on the uniform dataset. 
        model: tensorflow.keras.Model --- Final model trained.
    """
    
    efe = []
    acc = []
    efe_unif = []
    lab = (lab + 1)//2
    l_pts = [50, 200, 500, 1_000, 5_000, 10_000, 50_000, 200_000, 400_000, 
             600_000, 800_000, 1_000_000]
    p = np.log(100/5)/np.log(1/1000)
    a = 100/(1000**p)
    
    for pts in l_pts:
        model = keras.Sequential(
            [
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy())

        model.fit(data[:pts], lab[:pts], batch_size=32, epochs= int(round(a*pts**p, 2)), verbose=0)
        print("Computations for", pts, "points done")

        # Compute the metrics for the first model.
        efe.append(np.mean(bbmodelW(X_test) != np.where(model(X_test) > 0.5, 1, -1).flatten()))
        acc.append(np.mean(y_test == np.where(model(X_test) > 0.5, 1, 0).flatten()))
        efe_unif.append(np.mean(y_test_syn != np.where(model(data_test_syn) > 0.5, 1, -1).flatten()))
    
    return efe, acc, efe_unif, model

# This function finds the next power of two after the given number.
def next_power_of_2(x):
    """
    Compute the newt power of 2 after the given number.

    Arguments:
        x: int --- Integer whose next power of two we want to find.

    Returns:
        val: int --- Next power of 2.   
    """
    
    val = 1 if x == 0 else 2**(x - 1).bit_length()
    return val

# This function is used to normalize the given dataset.
# It is applied to the UCI high-dimensional datasets.
def normalize(X):
    """
    Normalize the columns of X by centering at the median and scaling based on quartiles.

    Arguments:
        X: numpy.ndarray --- Original dataset.

    Returns:
        X_scaled: numpy.ndarray --- Normalized dataset.
    """
    medians = np.median(X, axis=0)
    q1 = np.percentile(X, 0.25, axis=0)
    q3 = np.percentile(X, 99.75, axis=0)
    iqr = q3 - q1

    # Avoid division by zero
    iqr[iqr == 0] = 1.0

    X_centered = X - medians
    X_scaled = X_centered / (iqr / 2.0)

    return X_scaled

# This function is used to train the two stages of the Medium Neural Network copies.
# It is called during the Two-Stage Distance Copying Extension experiments.
# It returns snapshots of the results at different training dataset sizes. 
def train_copy_MNNd_2_stages (l_pts, data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train the two stages of MNN copies for different training dataset sizes.

    Arguments:
        l_pts: list --- List of dataset sizes at which training should be performed.
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns: 
        efed: list --- List of computed empirical fidelity errors for the first stage.
        accd: list --- List of the computed accuracies for the first stage.
        efe_unifd: list --- List of the computed empirical fidelity errors on the uniform dataset for the first stage. 
        modeld: tensorflow.keras.Model --- Final model trained for the first stage.
        efe: list --- List of computed empirical fidelity errors for the second stage.
        acc: list --- List of the computed accuracies for the second stage.
        efe_unif: list --- List of the computed empirical fidelity errors on the uniform dataset for the second stage. 
        model: tensorflow.keras.Model --- Final model trained for the second stage.
    """
    
    efe = []
    acc = []
    efe_unif = []

    efed = []
    accd = []
    efe_unifd = []

    p = np.log(100/5)/np.log(1/1000)
    a = 100/(1000**p)
    
    for pts in l_pts:
        # Initial model for distances
        modeld = keras.Sequential(
            [
                layers.Dense(128, activation = "relu"),
                layers.Dense(64, activation = "relu"),
                layers.Dense(32, activation = "relu"),
                layers.Dense(16, activation = "relu"),
                layers.Dense(1, activation = "linear"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        modeld.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        modeld.fit(data[:pts], lab[:pts], batch_size=32, epochs= int(round(a*pts**p, 2)), verbose=0)

        # Compute the metrics for the aux model.
        efed.append(np.mean(bbmodelW(X_test) != (np.sign(modeld(X_test)).flatten())))
        accd.append(np.mean((2*y_test-1) == (np.sign(modeld(X_test)).flatten())))
        efe_unifd.append(np.mean(y_test_syn != (np.sign(modeld(data_test_syn)).flatten())))
        
        # Data preparation
        n_pow2 = next_power_of_2(pts)
        sampler = qmc.Sobol(d=data.shape[1], scramble=False)
        sobol_points = sampler.random_base2(m=int(np.log2(n_pow2)))
        sobol_points = sobol_points[:pts]
        data_2s = 2 * sobol_points - 1

        # Copying with distances model
        model = keras.Sequential(
            [
                layers.Dense(128, activation = "relu"),
                layers.Dense(64, activation = "relu"),
                layers.Dense(32, activation = "relu"),
                layers.Dense(16, activation = "relu"),
                layers.Dense(1, activation = "linear"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        model.fit(data_2s, bbmodelW(data_2s)*(np.abs(modeld(data_2s)).flatten()), batch_size=32, epochs= int(round(a*pts**p, 2)), verbose=0)
        print("Computations for", pts, "points done")

        # Compute the metrics for the first model.
        efe.append(np.mean(bbmodelW(X_test) != (np.sign(model(X_test)).flatten())))
        acc.append(np.mean((2*y_test-1) == (np.sign(model(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model(data_test_syn)).flatten())))
    
    return efed, accd, efe_unifd, modeld, efe, acc, efe_unif, model

# This function is used to train the two stages of the Large Neural Network copies.
# It is called during the Two-Stage Distance Copying Extension experiments.
# It returns snapshots of the results at different training dataset sizes. 
def train_copy_LNNd_2_stages (l_pts, data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train the two stages of LNN copies for different training dataset sizes.
    
    Arguments:
        l_pts: list --- List of dataset sizes at which training should be performed.
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns: 
        efed: list --- List of computed empirical fidelity errors for the first stage.
        accd: list --- List of the computed accuracies for the first stage.
        efe_unifd: list --- List of the computed empirical fidelity errors on the uniform dataset for the first stage. 
        modeld: tensorflow.keras.Model --- Final model trained for the first stage.
        efe: list --- List of computed empirical fidelity errors for the second stage.
        acc: list --- List of the computed accuracies for the second stage.
        efe_unif: list --- List of the computed empirical fidelity errors on the uniform dataset for the second stage. 
        model: tensorflow.keras.Model --- Final model trained for the second stage.
    """
    
    efe = []
    acc = []
    efe_unif = []

    efed = []
    accd = []
    efe_unifd = []

    p = np.log(100/5)/np.log(1/1000)
    a = 100/(1000**p)
    
    for pts in l_pts:
        # Initial model for distances
        modeld = keras.Sequential(
            [
                layers.Dense(512, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="linear"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        modeld.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        modeld.fit(data[:pts], lab[:pts], batch_size=32, epochs= int(round(a*pts**p, 2)), verbose=0)

        # Compute the metrics for the aux model.
        efed.append(np.mean(bbmodelW(X_test) != (np.sign(modeld(X_test)).flatten())))
        accd.append(np.mean((2*y_test-1) == (np.sign(modeld(X_test)).flatten())))
        efe_unifd.append(np.mean(y_test_syn != (np.sign(modeld(data_test_syn)).flatten())))
        
        # Data preparation
        n_pow2 = next_power_of_2(pts)
        sampler = qmc.Sobol(d=data.shape[1], scramble=False)
        sobol_points = sampler.random_base2(m=int(np.log2(n_pow2)))
        sobol_points = sobol_points[:pts]
        data_2s = 2 * sobol_points - 1

        # Copying with distances model
        model = keras.Sequential(
            [
                layers.Dense(512, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(1, activation="linear"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        model.fit(data_2s, bbmodelW(data_2s)*(np.abs(modeld(data_2s)).flatten()), batch_size=32, epochs= int(round(a*pts**p, 2)), verbose=0)
        print("Computations for", pts, "points done")

        # Compute the metrics for the first model.
        efe.append(np.mean(bbmodelW(X_test) != (np.sign(model(X_test)).flatten())))
        acc.append(np.mean((2*y_test-1) == (np.sign(model(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model(data_test_syn)).flatten())))
    
    return efed, accd, efe_unifd, modeld, efe, acc, efe_unif, model


# This function is used to train the two stages of the Small Neural Network copies.
# It is called during the Two-Stage Distance Copying Extension experiments.
# It returns snapshots of the results at different training dataset sizes. 
def train_copy_SNNd_2_stages (l_pts, data, lab, X_test, y_test, data_test_syn, y_test_syn, bbmodelW):
    """
    Train the two stages of SNN copies for different training dataset sizes.
    
    Arguments:
        l_pts: list --- List of dataset sizes at which training should be performed.
        data: numpy.ndarry --- Dataset where the copies will be trained.
        lab: numpy.ndarry --- Targets of the dataset where the copies will be trained.
        X_test: numpy.ndarry --- Test dataset to compute performance metrics.
        y_test: numpy.ndarry --- Targets of the test dataset.
        data_test_syn: numpy.ndarry --- Synthetic uniform dataset to compute performance metrics.
        y_test_syn: numpy.ndarry --- Targets of the synthetic uniform dataset.
        bbmodelW: FunctionType --- Black box model wrapped in a custom Python function. 

    Returns: 
        efed: list --- List of computed empirical fidelity errors for the first stage.
        accd: list --- List of the computed accuracies for the first stage.
        efe_unifd: list --- List of the computed empirical fidelity errors on the uniform dataset for the first stage. 
        modeld: tensorflow.keras.Model --- Final model trained for the first stage.
        efe: list --- List of computed empirical fidelity errors for the second stage.
        acc: list --- List of the computed accuracies for the second stage.
        efe_unif: list --- List of the computed empirical fidelity errors on the uniform dataset for the second stage. 
        model: tensorflow.keras.Model --- Final model trained for the second stage.
    """
    
    efe = []
    acc = []
    efe_unif = []

    efed = []
    accd = []
    efe_unifd = []

    p = np.log(100/5)/np.log(1/1000)
    a = 100/(1000**p)
    
    for pts in l_pts:
        # Initial model for distances
        modeld = keras.Sequential(
            [
                layers.Dense(32, activation = "relu"),
                layers.Dense(16, activation = "relu"),
                layers.Dense(1, activation = "linear"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        modeld.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        modeld.fit(data[:pts], lab[:pts], batch_size=32, epochs= int(round(a*pts**p, 2)), verbose=0)

        # Compute the metrics for the aux model.
        efed.append(np.mean(bbmodelW(X_test) != (np.sign(modeld(X_test)).flatten())))
        accd.append(np.mean((2*y_test-1) == (np.sign(modeld(X_test)).flatten())))
        efe_unifd.append(np.mean(y_test_syn != (np.sign(modeld(data_test_syn)).flatten())))
        
        # Data preparation
        n_pow2 = next_power_of_2(pts)
        sampler = qmc.Sobol(d=data.shape[1], scramble=False)
        sobol_points = sampler.random_base2(m=int(np.log2(n_pow2)))
        sobol_points = sobol_points[:pts]
        data_2s = 2 * sobol_points - 1

        # Copying with distances model
        model = keras.Sequential(
            [
                layers.Dense(32, activation = "relu"),
                layers.Dense(16, activation = "relu"),
                layers.Dense(1, activation = "linear"),
            ]
        )
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer="adam", loss=keras.losses.MeanSquaredError())
        model.fit(data_2s, bbmodelW(data_2s)*(np.abs(modeld(data_2s)).flatten()), batch_size=32, epochs= int(round(a*pts**p, 2)), verbose=0)
        print("Computations for", pts, "points done")

        # Compute the metrics for the first model.
        efe.append(np.mean(bbmodelW(X_test) != (np.sign(model(X_test)).flatten())))
        acc.append(np.mean((2*y_test-1) == (np.sign(model(X_test)).flatten())))
        efe_unif.append(np.mean(y_test_syn != (np.sign(model(data_test_syn)).flatten())))
    
    return efed, accd, efe_unifd, modeld, efe, acc, efe_unif, model

# This function loads the models trained in the Two-Stage Distance Copying Extension and evaluates the errors of the distances they predict.
# The results are averaged over all seeds and then printed.
# In addition, it also plots scatter plots that show the relationship between the real and predicted distances, for the specified seed. 
def evaluate_distance_prediction(folder_path, file_prefix, synthetic_data, model_stage=2, num_rdc_samples=10000, seed_plot_index=42):
    """ 
    Compute, print and plot results that compare the real and predicted distances for the corresponding distance-based copy (Two-Stage Extension).

    Arguments:
        folder_path: str --- Path to the folder where results are stored.
        file_prefix: str --- Name of the results file truncated to remove the seed number. 
        synthetic_data: numpy.ndarray --- Synthetic data points where predicted distances will be evaluated.
        model_stage: int --- A value equal to 1 or 2, aimed at selecting the stage 1 or stage 2 model.
        num_rdc_samples: int --- Number of cloud points to use in the computation of ground truth distances.
        seed_plot_index: int --- Seed index to use for the scatter plot.
    
    Returns:
        Nothing
    """
    dim = synthetic_data.shape[1]
    
    seed_files = [f for f in os.listdir(folder_path) if f.startswith(file_prefix)]
    if not seed_files:
        raise FileNotFoundError(f"No files starting with '{file_prefix}' found in {folder_path}")

    maes = []
    rmses = []

    predicted_all = []
    true_all = []
    
    for fname in sorted(seed_files):
        mat = re.search(r'seed(\d+)', fname)
        if not mat:
            print(f"Skipping file without seed: {fname}")
            continue
        seed_idx = int(mat.group(1))

        file_path = os.path.join(folder_path, fname)
        with open(file_path, "rb") as f:
            data_loaded = pickle.load(f)

        # Load selected surrogate model
        model_key = f"model{model_stage}"
        if model_key not in data_loaded:
            continue
        model_entry = data_loaded[model_key]
        if "model" not in model_entry:
            continue
        model = model_entry["model"]

        # Load black-box model
        if "blackb" not in data_loaded or "model" not in data_loaded["blackb"]:
            continue
        bbmodel = data_loaded["blackb"]["model"]

        def bbmodelW(x):
            if isinstance(bbmodel, tf.keras.models.Model):
                return np.where(bbmodel(x) > 0.5, 1, -1).flatten()
            return np.where(bbmodel.predict(x) > 0.5, 1, -1).flatten()

        # Generate directions for distance estimation
        rdc, uni = generate(dim, num_rdc_samples)

        # Estimate true distances using cdistance
        true_dists = []
        for x in synthetic_data:
            dist = cdistance(7, x, rdc.T, uni, bbmodelW, 3)
            true_dists.append(dist)
        true_dists = np.array(true_dists)

        # Predict distances using surrogate model
        predicted_dists = model(synthetic_data).numpy().flatten()
        predicted_dists = np.abs(predicted_dists)

        # Compute error metrics
        mae = mean_absolute_error(true_dists, predicted_dists)
        rmse = root_mean_squared_error(true_dists, predicted_dists)

        maes.append(mae)
        rmses.append(rmse)

        # Store data for plotting
        if seed_idx == seed_plot_index:
            predicted_all = predicted_dists
            true_all = true_dists

    # Aggregate metrics
    mae_mean = np.mean(maes)
    mae_std = np.std(maes)
    rmse_mean = np.mean(rmses)
    rmse_std = np.std(rmses)

    # Print summary
    print("\n--- Final Distance Prediction Evaluation ---")
    print(f"Model Stage: {model_stage}")
    print(f"Mean MAE : {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"Mean RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")

    # Plot predictions vs. true distances for selected seed
    if len(predicted_all) > 0 and len(true_all) > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(true_all, predicted_all, c='blue', label='Predictions')
        max_val = max(max(true_all), max(predicted_all))
        plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal')
        plt.xlabel("True Distance to Decision Boundary")
        plt.ylabel("Predicted Distance")
        plt.title(f"Model stage {model_stage} - Seed {seed_plot_index}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# This function loads the models trained in Experiment 1 and evaluates the errors of the distances they predict.
# The results are printed in a table, that averages them over all seeds.
# In addition, it also plots scatter plots that show the relationship between the real and predicted distances, for the specified seed. 
def evaluate_distance_prediction_general(folder_path, file_prefix, test_data, synthetic_data, model_stage=2, num_rdc_samples=10000, seed_plot_index=42):
    """ 
    Compute, print and plot results that compare the real and predicted distances for the corresponding distance-based copy (Experiment 3).

    Arguments:
        folder_path: str --- Path to the folder where results are stored.
        file_prefix: str --- Name of the results file truncated to remove the type of copy and the seed number. 
        test_data: numpy.ndarray --- Test data points where predicted distances will be evaluated.
        synthetic_data: numpy.ndarray --- Synthetic data points where predicted distances will be evaluated.
        model_stage: int --- A value equal to 1 or 2, aimed at selecting the Algorithm 1 or Algorithm 2 copies.
        num_rdc_samples: int --- Number of cloud points to use in the computation of ground truth distances.
        seed_plot_index: int --- Seed index to use for the scatter plot.

    Returns:
        Nothing.
    """
    # --------------------------------------------------------
    # Settings
    # --------------------------------------------------------
    model_tags = ["SNN", "MNN", "LNN", "GB"]
    model_suffixes = [1, 2, 3, 4]
    datasets = [test_data, synthetic_data]
    dataset_labels = ["Test Data", "Synthetic Data"]

    # Table storage
    results_table = []

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # --------------------------------------------------------
    # Iterate over two datasets (rows)
    # --------------------------------------------------------
    for row, (data_row, data_label) in enumerate(zip(datasets, dataset_labels)):

        dim = data_row.shape[1]

        # --------------------------------------------------------
        # Iterate over four model variants (columns)
        # --------------------------------------------------------
        for col, (m_suffix, m_tag) in enumerate(zip(model_suffixes, model_tags)):

            # Build prefix: e.g. "results_DS4_2_1_seed"
            full_prefix = f"{file_prefix}_{m_suffix}_seed"

            # Find files
            seed_files = [f for f in os.listdir(folder_path) if f.startswith(full_prefix)]
            if not seed_files:
                raise FileNotFoundError(
                    f"No files starting with '{full_prefix}' found in {folder_path}"
                )

            maes = []
            rmses = []
            predicted_all = []
            true_all = []

            # --------------------------------------------------------
            # Loop over seed files
            # --------------------------------------------------------
            for fname in sorted(seed_files):
                mat = re.search(r'seed(\d+)', fname)
                if not mat:
                    continue
                seed_idx = int(mat.group(1))

                file_path = os.path.join(folder_path, fname)
                with open(file_path, "rb") as f:
                    data_loaded = pickle.load(f)

                # Load surrogate model
                model_key = f"model{model_stage}"
                if model_key not in data_loaded:
                    continue
                model_entry = data_loaded[model_key]
                if "model" not in model_entry:
                    continue
                model = model_entry["model"]

                # Load black-box model
                if "blackb" not in data_loaded or "model" not in data_loaded["blackb"]:
                    continue
                bbmodel = data_loaded["blackb"]["model"]

                def bbmodelW(x):
                    if isinstance(bbmodel, tf.keras.models.Model):
                        return np.where(bbmodel(x) > 0.5, 1, -1).flatten()
                    return np.where(bbmodel.predict(x) > 0.5, 1, -1).flatten()

                # Directions
                rdc, uni = generate(dim, num_rdc_samples)

                # -------- True distances
                true_dists = []
                for x in data_row:
                    dist = cdistance(7, x, rdc.T, uni, bbmodelW, 3)
                    true_dists.append(dist)
                true_dists = np.array(true_dists)

                # -------- Predicted distances
                if isinstance(model, types.FunctionType):
                    predicted_dists = model(data_row).numpy().flatten()
                elif isinstance(model, tf.keras.Model):
                    predicted_dists = model(data_row).numpy().flatten()
                else:
                    predicted_dists = model.predict(data_row).flatten()

                predicted_dists = np.abs(predicted_dists)

                # Metrics
                mae = mean_absolute_error(true_dists, predicted_dists)
                rmse = root_mean_squared_error(true_dists, predicted_dists)

                maes.append(mae)
                rmses.append(rmse)

                # Save plotting data
                if seed_idx == seed_plot_index:
                    predicted_all = predicted_dists
                    true_all = true_dists

            # Aggregate metrics
            mae_mean = np.mean(maes)
            mae_std = np.std(maes)
            rmse_mean = np.mean(rmses)
            rmse_std = np.std(rmses)

            # Store in table
            results_table.append({
                "Dataset": data_label,
                "Model": m_tag,
                "MAE_mean": mae_mean,
                "MAE_std": mae_std,
                "RMSE_mean": rmse_mean,
                "RMSE_std": rmse_std
            })

            # Plot
            ax = axes[row, col]
            if len(predicted_all) > 0:
                ax.scatter(true_all, predicted_all, c='blue', s=10)
                max_val = max(max(true_all), max(predicted_all))
                ax.plot([0, max_val], [0, max_val], 'r--')

            ax.set_title(f"{data_label} – {m_tag}")
            ax.set_xlabel("True Distance")
            ax.set_ylabel("Predicted Distance")
            ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Convert results table to DataFrame and print it
    df_results = pd.DataFrame(results_table)
    print("\n=== Final Distance Prediction Results ===")
    print(df_results.to_string(index=False))


# This functions loads the results of the corresponding experiment.
# Then, it computes the means of the final metrics across the different seeds available.
# Finally, these means are returned in a convenient list of tuples.
def retrieve_results_final_stats(folder_path, file_prefix_pattern):
    """
    Loads the results for all DS and copy combinations in the folder matching the prefix pattern and computes their final metrics.

    Arguments:
        folder_path: str --- Path to the folder where results are stored.
        file_prefix_pattern: str --- Name of the results file truncated to remove the dataset, models and seed numbers.
    
    Returns:
        results:list --- List of tuples describing the algorithm, models and dataset together with their mean final metrics.
    """

    # List all files in the folder
    all_files = os.listdir(folder_path)
    matching_files = [f for f in all_files if f.startswith(file_prefix_pattern)]

    # Group files by DS and copy numbers
    pattern = re.compile(r'(DS\d+_\d+_\d+)_seed')
    groups = {}
    for f in matching_files:
        match = pattern.search(f)
        if match:
            group_key = match.group(1)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(f)

    results = []

    # Mapping for algorithm names
    algo_map = {
        "Algorithm 1 Copy": "Algo. 1",
        "Algorithm 2 Copy": "Algo. 2",
        "Hard Copy": "Hard"
    }

    # Mapping for DS copy numbers
    model_map = {1: "RF", 2: "GB", 3: "NN"}
    network_map = {1: "SNN", 2: "MNN", 3: "LNN", 4: "GB"}

    length = len(groups.items())
    ii = 1
    # Retrieve results
    for group_key, files in groups.items():
        ds_match = re.search(r'DS(\d+)', group_key)
        ds_number = int(ds_match.group(1)) if ds_match else None

        copy_match = re.search(r'_(\d+)_(\d+)$', group_key)
        if copy_match:
            model_num = int(copy_match.group(1))
            network_num = int(copy_match.group(2))
            model_network = f"{model_map.get(model_num, 'Unknown')}/{network_map.get(network_num, 'Unknown')}"
        else:
            model_network = "Unknown"

        # Build full prefix for load_all_seeds
        full_prefix = file_prefix_pattern + group_key.split("DS")[1] + "_seed"
        model_data, blackb_acc = load_all_seeds(folder_path, full_prefix)

        # Compute final stats using the corresponding function
        final_stats_df = compute_final_stats(model_data, blackb_acc)

        # Extract acc and efe_unif means for each model
        for model_id, label in [(1, "Algorithm 1 Copy"), (2, "Algorithm 2 Copy"), (3, "Hard Copy")]:
            row = final_stats_df.loc[final_stats_df["model"] == f"Model {model_id}"]
            if not row.empty:
                acc_mean = float(row["acc m."].values[0])
                efe_unif_mean = float(row["efe_unif m."].values[0])
            else:
                acc_mean = None
                efe_unif_mean = None
                    
            results.append((algo_map[label], model_network, ds_number, acc_mean, efe_unif_mean))
        
        # Completion counter
        if ii%(int(length/5)) == 0:
            print(f"Computations done: {ii} out of {length}")
        ii+=1
    
    print ("All results retrieved")
    
    return results

# This functions loads the results of the corresponding experiment.
# Then, it computes the means of the average metrics across the different seeds available.
# To do it, it uses the previous average_metric_interpolated function.
def retrieve_results_avg(folder_path, file_prefix_pattern):
    """
    Loads the results for all DS and copy combinations in the folder matching the prefix pattern and computes their average metrics.

    Arguments:
        folder_path: str --- Path to the folder where results are stored.
        file_prefix_pattern: str --- Name of the results file truncated to remove the dataset, models and seed numbers.
    
    Returns:
        results:list --- List of tuples describing the algorithm, models and dataset together with their mean average metrics.
    """

    # List all files in the folder
    all_files = os.listdir(folder_path)
    matching_files = [f for f in all_files if f.startswith(file_prefix_pattern)]

    # Group files by DS and copy numbers
    pattern = re.compile(r'(DS\d+_\d+_\d+)_seed')
    groups = {}
    for f in matching_files:
        match = pattern.search(f)
        if match:
            group_key = match.group(1)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(f)

    results = []

    # Mapping for algorithm names
    algo_map = {
        "Algorithm 1 Copy": "Algo. 1",
        "Algorithm 2 Copy": "Algo. 2",
        "Hard Copy": "Hard"
    }

    # Mapping for DS copy numbers
    model_map = {1: "RF", 2: "GB", 3: "NN"}
    network_map = {1: "SNN", 2: "MNN", 3: "LNN", 4: "GB"}

    length = len(groups.items())

    ii = 1

    # Retrieve results
    for group_key, files in groups.items():
        ds_match = re.search(r'DS(\d+)', group_key)
        ds_number = int(ds_match.group(1)) if ds_match else None

        copy_match = re.search(r'_(\d+)_(\d+)$', group_key)
        if copy_match:
            model_num = int(copy_match.group(1))
            network_num = int(copy_match.group(2))
            model_network = f"{model_map.get(model_num, 'Unknown')}/{network_map.get(network_num, 'Unknown')}"
        else:
            model_network = "Unknown"

        # Build the full prefix for load_all_seeds
        full_prefix = file_prefix_pattern + group_key.split("DS")[1] + "_seed"
        model_data, _ = load_all_seeds(folder_path, full_prefix)

        for model_id, label in [(1, "Algorithm 1 Copy"), (2, "Algorithm 2 Copy"), (3, "Hard Copy")]:
            data = model_data[model_id]

            # Compute metrics
            x_pts_acc, y_pts_acc = average_metric_interpolated(data["pts"], data["acc"], log_scale=True)
            x_pts_efe, y_pts_efe = average_metric_interpolated(data["pts"], data["efe_unif"], log_scale=True)

            acc_value = np.mean(np.nan_to_num(y_pts_acc, nan=0.0)) if y_pts_acc is not None else None
            efe_value = np.mean(np.nan_to_num(y_pts_efe, nan=0.0)) if y_pts_efe is not None else None

            results.append((algo_map[label], model_network, ds_number, acc_value, efe_value))

        # Completion counter
        if ii%(int(length/5)) == 0:
            print(f"Computations done: {ii} out of {length}")
        ii+=1
    
    print ("All results retrieved")
    
    return results

# This functions computes the mean of the metrics achieved in a certain dataset by a certain algorithm or method. 
def get_mean(algo, datasets, data):
    """
    Compute the means across model combinations of the provided metrics for the specified datasets and algorithm.

    Arguments:
        algo: str --- Name of the algorithm for which we compute the metric means. 
        datasets: range --- Range of datasets for which to compute the means.
        data: collections.defaultdict --- Custom dictionary indexed by tuples that contains the metrics to be averaged.

    Returns:
        arr_mean_acc: numpy.ndarry --- Array of the mean accuracies for the specified datasets and algorithm.
        arr_mean_1fid: numpy.ndarry --- Array of the mean 1-fidelity error for the specified datasets and algorithm.
    """
    
    mean_acc, mean_1fid = [], []

    # Compute the means across models for the corresponding algorithm and datasets
    for ds in datasets:
        if (algo, ds) in data:
            acc = np.mean(data[(algo, ds)]['accuracy'])
            err = np.mean(data[(algo, ds)]['error'])
            mean_acc.append(acc)
            mean_1fid.append(1 - err)
        else:
            mean_acc.append(np.nan)
            mean_1fid.append(np.nan)

    arr_mean_acc = np.array(mean_acc)
    arr_mean_1fid = np.array(mean_1fid)
    
    return arr_mean_acc, arr_mean_1fid

# This function is used to plot with bar charts the results from Experiment 1.
# These results are averaged across all model combinations and shown for each dataset and algorithm.
# The results of the hard copy are used as the 0 x-axis.
def plot_algo_comparison(results):
    """
    Compute the means of the provided metrics of Experiment 1 across all model combinations and plot them with bar charts.

    Arguments:
        results:list --- List of tuples describing the algorithm, models and dataset together with their mean metrics.

    Returns:
        Nothing
    """
    
    # Filter out rows ending with '/GB'
    filtered_results = [t for t in results if not t[1].endswith('/GB')]

    # Organize the data
    data = defaultdict(lambda: {'accuracy': [], 'error': []})
    for algo, model, dataset, acc, err in filtered_results:
        data[(algo, dataset)]['accuracy'].append(acc)
        data[(algo, dataset)]['error'].append(err)

    datasets = range(1, 7)

    # Compute the means
    hard_acc, hard_1fid = get_mean('Hard', datasets, data)
    algo1_acc, algo1_1fid = get_mean('Algo. 1', datasets, data)
    algo2_acc, algo2_1fid = get_mean('Algo. 2', datasets, data)

    # Compute differences relative to Hard
    algo1_acc_diff = algo1_acc - hard_acc
    algo2_acc_diff = algo2_acc - hard_acc
    algo1_1fid_diff = algo1_1fid - hard_1fid
    algo2_1fid_diff = algo2_1fid - hard_1fid

    # Plot
    bar_width = 0.15
    x = np.arange(len(datasets))
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, ds in enumerate(datasets):
        ax.bar(x[i] + offsets[0], algo1_acc_diff[i], width=bar_width, label=r'Alg. 1 $\mathcal{A}_{\mathcal{C}}$' if i==0 else "", color='tab:red')
        ax.bar(x[i] + offsets[1], algo2_acc_diff[i], width=bar_width, label=r'Alg. 2 $\mathcal{A}_{\mathcal{C}}$' if i==0 else "", color='tab:blue')
        ax.bar(x[i] + offsets[2], algo1_1fid_diff[i], width=bar_width, label=r'Alg. 1 $1-R_{\mathrm{emp}}^{\mathcal{S}}$' if i==0 else "", color='tab:red', alpha=0.5, hatch='//')
        ax.bar(x[i] + offsets[3], algo2_1fid_diff[i], width=bar_width, label=r'Alg. 2 $1-R_{\mathrm{emp}}^{\mathcal{S}}$' if i==0 else "", color='tab:blue', alpha=0.5, hatch='//')

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Dataset {d}' for d in datasets])
    ax.set_ylabel('Difference relative to Hard copy')
    ax.set_title('Performance of Algo.1 cp. and Algo.2 cp. relative to Hard cp. (symlog scale)')
    ax.legend(ncol=2)
    ax.grid(True, axis='y')
    ax.set_yscale('symlog', linthresh=0.001)

    plt.tight_layout()
    plt.show()

# This function is used to plot with bar charts the results from the Extension experiment.
# These results are averaged across all model combinations and shown for each dataset and copy stage.
# The results of the hard copy are used as the 0 x-axis.
def plot_algo_comparison_extension(results):
    """
    Compute the means of the provided metrics of the Extension experiment across all model combinations and plot them with bar charts.

    Arguments:
        results:list --- List of tuples describing the algorithm, models and dataset together with their mean metrics.

    Returns:
        Nothing
    """
    
    # Filter out rows ending with '/GB'
    filtered_results = [t for t in results if not t[1].endswith('/GB')]

    # Organize data
    data = defaultdict(lambda: {'accuracy': [], 'error': []})
    for algo, model, dataset, acc, err in filtered_results:
        data[(algo, dataset)]['accuracy'].append(acc)
        data[(algo, dataset)]['error'].append(err)

    datasets = range(4, 7)

    # Compute the means
    hard_acc, hard_1fid = get_mean('Hard', datasets, data)
    algo1_acc, algo1_1fid = get_mean('Algo. 1', datasets, data)
    algo2_acc, algo2_1fid = get_mean('Algo. 2', datasets, data)

    # Compute differences relative to Hard
    algo1_acc_diff = algo1_acc - hard_acc
    algo2_acc_diff = algo2_acc - hard_acc
    algo1_1fid_diff = algo1_1fid - hard_1fid
    algo2_1fid_diff = algo2_1fid - hard_1fid

    # Plot
    bar_width = 0.15
    x = np.arange(len(datasets))
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, ds in enumerate(datasets):
        ax.bar(x[i] + offsets[0], algo1_acc_diff[i], width=bar_width, label=r'St. 1 $\mathcal{A}_{\mathcal{C}}$' if i==0 else "", color='tab:blue')
        ax.bar(x[i] + offsets[1], algo2_acc_diff[i], width=bar_width, label=r'St. 2 $\mathcal{A}_{\mathcal{C}}$' if i==0 else "", color='tab:green')
        ax.bar(x[i] + offsets[2], algo1_1fid_diff[i], width=bar_width, label=r'St. 1 $1-R_{\mathrm{emp}}^{\mathcal{S}}$' if i==0 else "", color='tab:blue', alpha=0.5, hatch='//')
        ax.bar(x[i] + offsets[3], algo2_1fid_diff[i], width=bar_width, label=r'St. 2 $1-R_{\mathrm{emp}}^{\mathcal{S}}$' if i==0 else "", color='tab:green', alpha=0.5, hatch='//')

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Dataset {d}' for d in datasets])
    ax.set_ylabel('Difference relative to Hard copy')
    ax.set_title('Performance of St.1 cp. and St.2 cp. relative to Hard cp.')
    ax.legend(ncol=2)
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.show()