from pred_elem_seq import (
    get_metrics,
    get_model_preds_prqt,
    ConvLinNet,
)
from pathlib import Path
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
import re
import math
import pandas as pd

# import pvlib
# import rasterio


def round_to_multiple(number, multiple, direction="nearest"):
    digits = "%e" % multiple
    rnd_places = int(digits.split("e")[1])
    if rnd_places > 0:
        rnd_places = 0
    else:
        rnd_places = abs(rnd_places)

    if direction == "nearest":
        return round(multiple * round(number / multiple), rnd_places)
    elif direction == "up":
        return round(multiple * math.ceil(number / multiple), rnd_places)
    elif direction == "down":
        return round(multiple * math.floor(number / multiple), rnd_places)
    else:
        return round(multiple * round(number / multiple), rnd_places)


def get_low_high_div_rnd(
    min_v, max_v, n_div, fixed_low=False, fixed_high=False, ret_int=False
):
    # get lower bound, upper bound, spacing, and rounding places for plot axes (i.e., tick labels)
    # based on the minimum and maximum values of the series as well as the number
    # of desired tick divisions.
    # prioritize tick labels that fall on nice numbers
    # possibility of fixing lower and upper bound at minimum and maximum series values, respectively

    # cast integer into floats
    if isinstance(min_v, int):
        min_v = float(min_v)

    if isinstance(max_v, int):
        max_v = float(max_v)

    # changes = [1, -1, 2, -2]
    changes = [0, -1, 1, -2, 2]

    final_low_list = []
    final_high_list = []
    final_div_list = []
    final_diff_list = []

    num_div_list = [n_div + c for c in changes if n_div + c >= 4]

    for i_n, num_div in enumerate(num_div_list):
        low_list = []
        high_list = []
        div_list = []

        div = (max_v - min_v) / (num_div - 2)
        digits = "%e" % div

        lead_digit = float(digits.split("e")[0])
        exp = int(digits.split("e")[1])

        if lead_digit >= 6.5:
            lead_digit = 8
        elif 6.5 > lead_digit >= 4.5:
            lead_digit = 5
        elif 4.5 > lead_digit >= 3.5:
            lead_digit = 4
        elif 3.5 > lead_digit >= 2.5:
            lead_digit = 3
        elif 2.5 > lead_digit >= 1.5:
            lead_digit = 2
        else:
            lead_digit = 1

        # try moving up and then down the list of possible leading digits
        dig_list = [1, 2, 3, 4, 5, 8]

        i_d = dig_list.index(lead_digit)
        for chg_1 in changes:
            for chg_2 in changes:
                i_1 = (i_d + chg_1) % len(dig_list)
                if i_d + chg_1 > len(dig_list) - 1:
                    e_1 = exp + 1
                elif i_d + chg_1 < 0:
                    e_1 = exp - 1
                else:
                    e_1 = exp

                d_1 = float(f"{dig_list[i_1]}e{e_1}")

                i_2 = (i_d + chg_2) % len(dig_list)
                if i_d + chg_2 > len(dig_list) - 1:
                    e_2 = exp + 1
                elif i_d + chg_2 < 0:
                    e_2 = exp - 1
                else:
                    e_2 = exp

                d_2 = float(f"{dig_list[i_2]}e{e_2}")

                if d_1 == d_2:
                    low_sub = [d_1, 0, 0, d_1]
                    high_add = [d_1, d_1, 0, 0]
                    ticks_try = [d_1, d_1, d_1, d_1]
                    e_try = [e_1, e_1, e_1, e_1]

                else:
                    low_sub = [
                        d_2,
                        d_2,
                        d_2,
                        d_2,
                        d_1,
                        d_1,
                        d_1,
                        d_1,
                        d_1,
                        d_1,
                        d_2,
                        d_2,
                        0,
                        0,
                        0,
                        0,
                    ]
                    high_add = [
                        d_2,
                        d_2,
                        d_1,
                        d_1,
                        d_2,
                        d_2,
                        d_1,
                        d_1,
                        0,
                        0,
                        0,
                        0,
                        d_1,
                        d_1,
                        d_2,
                        d_2,
                    ]
                    ticks_try = [
                        d_2,
                        d_1,
                        d_2,
                        d_1,
                        d_2,
                        d_1,
                        d_2,
                        d_1,
                        d_1,
                        d_2,
                        d_1,
                        d_2,
                        d_1,
                        d_2,
                        d_1,
                        d_2,
                    ]
                    e_try = [
                        e_2,
                        e_1,
                        e_2,
                        e_1,
                        e_2,
                        e_1,
                        e_2,
                        e_1,
                        e_1,
                        e_2,
                        e_1,
                        e_2,
                        e_1,
                        e_2,
                        e_1,
                        e_2,
                    ]

                for k in range(len(low_sub)):
                    rnd_plc = -min(0, e_try[k])

                    if fixed_low:
                        low = min_v
                    else:
                        low = round(
                            round_to_multiple(min_v, ticks_try[k], "down") - low_sub[k],
                            rnd_plc,
                        )

                    if fixed_high:
                        high = max_v
                    else:
                        high = round(
                            round_to_multiple(max_v, ticks_try[k], "up") + high_add[k],
                            rnd_plc,
                        )

                    if rnd_plc == 0:
                        ticks = (high - low) / ticks_try[k] + 1
                    else:
                        ticks = round((high - low) / ticks_try[k] + 1, rnd_plc)

                    if ticks.is_integer() and int(ticks) == num_div:
                        low_list.append(low)
                        high_list.append(high)
                        div_list.append(ticks_try[k])

        # now select the triplet (low, high, div) where the division has the least number of decimal places
        # if tied, choose the first one in the list
        dcm_places = np.array(
            [
                sum(
                    [
                        0 if f.is_integer() else str(f)[::-1].find(".")
                        for f in [low_list[i], high_list[i], div_list[i]]
                    ]
                )
                for i in range(len(low_list))
            ]
        )

        if dcm_places.size > 0:
            min_idx = np.where(dcm_places == dcm_places.min())
            low_list_min = np.array(low_list)[min_idx]
            high_list_min = np.array(high_list)[min_idx]
            div_list_min = np.array(div_list)[min_idx]

            # finally select the smallest range
            spread = high_list_min - low_list_min
            final_idx = np.argmin(spread)
            final_low_list.append(low_list_min[final_idx])
            final_high_list.append(high_list_min[final_idx])
            final_div_list.append(div_list_min[final_idx])
            final_diff_list.append(
                (final_high_list[-1] - final_low_list[-1]) / (max_v - min_v)
            )

    # take the final limits that are closest
    min_diff_idx = np.argmin(np.array(final_diff_list))
    final_low = final_low_list[min_diff_idx]
    final_high = final_high_list[min_diff_idx]
    final_div = final_div_list[min_diff_idx]

    # check if division is an integer
    if ret_int and final_div.is_integer():
        final_low = int(final_low)
        final_high = int(final_high)
        final_div = int(final_div)
        rnd_places = 0

    elif not ret_int and final_div.is_integer():
        final_low = int(final_low)
        final_high = int(final_high)
        final_div = int(final_div)
        rnd_places = 0

    elif not ret_int and not final_div.is_integer():
        div_plcs = str(final_div)[::-1].find(".")
        low_plcs = str(final_low)[::-1].find(".")
        high_plcs = str(final_high)[::-1].find(".")
        rnd_places = max(div_plcs, low_plcs, high_plcs)

        final_low = np.round(final_low, rnd_places)
        final_high = np.round(final_high, rnd_places)
        final_div = np.round(final_div, rnd_places)

    return final_low, final_high, final_div, rnd_places


def plot_onecyclelr_schedule(
    ann, epochs=15, max_lr=2e-4, pct_start=0.3, div_f_lr=2, fin_div_f_lr=4
):
    # create dummy model
    ann_model = ConvLinNet(ann)
    # get optimizer with minimum learn rate
    optimizer = torch.optim.AdamW(
        ann_model.parameters(),
        lr=pct_start,
    )

    # n_lhs = 53**2
    # train_perc = 0.7
    # shuffle_group = 8
    # steps_per_epoch = shuffle_group * round(train_perc * n_lhs * 8760 / shuffle_group)
    steps_per_epoch = 100

    # get OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=pct_start,
        div_factor=div_f_lr,
        final_div_factor=fin_div_f_lr,
    )
    # instantiate learn rate and training row arrays to fill in with for loop
    lr = np.zeros(steps_per_epoch * epochs)
    rows = np.zeros(steps_per_epoch * epochs)
    # fill in arrays with values from scheduler
    for i in range(len(rows)):
        lr[i] = scheduler.get_lr()[0]
        rows[i] = i / steps_per_epoch
        scheduler.step()
    print(f"final lr: {lr[-1]}")
    # make plot showing evolution of learn rate with training rows
    font_reg = FontProperties(weight="normal", size=8)
    font_bold = FontProperties(weight="bold", size=8)
    matplotlib.rcParams["font.size"] = 8
    matplotlib.rcParams["font.family"] = "sans-serif"

    margins = 20
    line_width = 210 - margins * 2

    fig, ax = plt.subplots(1, 1, figsize=(line_width / 3 / 25.4, line_width / 3 / 25.4))
    ax.plot(rows, lr)
    yticks = [0.000025, 0.0001, 0.0002]
    y_ticklabels = ["0.000025", "0.0001", "0.0002"]
    ax.set_yticks(
        yticks,
        y_ticklabels,
    )
    ax.set_yticklabels(y_ticklabels, fontproperties=font_reg)
    ax.set_ylim([0.0000205, 0.000205])
    xticks = range(0, epochs + 1, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontproperties=font_reg)
    ax.set_xlim([0, epochs])
    # ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=True)
    ax.set_ylabel("Learning rate", fontproperties=font_bold)
    ax.set_xlabel("Epochs", fontproperties=font_bold)
    ax.grid()

    fig.savefig(
        Path(
            f"pred_elem_seq/datafiles/figures/learn_rate/onecyclelr scheduler - max_lr_{max_lr}.pdf"
        ),
        bbox_inches="tight",
        pad_inches=0,
    )


def get_r2(y_pred, y_true):
    return 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()


def set_ticks_labels(ax, i, low, high, div, rnd_places):
    ax.set_yticks(np.arange(low, high + div, div).round(rnd_places))
    ax.set_ylim([low, high])
    ax.set_xticks(np.arange(low, high + div, div).round(rnd_places))
    ax.set_xlim([low, high])
    ax.grid()

    if i == 0:
        # set y tick labels for leftmost column of plots
        ax.ticklabel_format(
            style="sci", axis="both", scilimits=(0, 0), useMathText=True
        )
        ax.xaxis.offsetText.set_fontsize(8)
        ax.yaxis.offsetText.set_fontsize(8)

    elif i > 0:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0), useMathText=True)
        ax.xaxis.offsetText.set_fontsize(8)
        ax.yaxis.get_major_formatter().set_scientific(False)
        # remove y tick labels for second and third column of plots
        for tic in ax.yaxis.get_major_ticks():
            # tic.tick1line.set_visible(False)
            # tic.tick2line.set_visible(False)
            tic.label1.set_visible(False)
            tic.label2.set_visible(False)
        for tic in ax.yaxis.get_minor_ticks():
            # tic.tick1line.set_visible(False)
            # tic.tick2line.set_visible(False)
            tic.label1.set_visible(False)
            tic.label2.set_visible(False)

    ax.tick_params(axis="both", which="major", labelsize=8)


def time_of_day_errors(ann_model_path, ann_datasets, run_ann):
    # get model predictions
    res = get_model_preds_prqt(
        ann_model_path, ann_datasets=ann_datasets, run_ann=run_ann
    )
    # calculate absolute errors
    error_abs = np.abs(res["labels"] - res["preds"])
    # create dataframe with absolute errors and time index
    df = pd.DataFrame(
        error_abs,
        columns=["error heat [kWh]", "error cool [kWh]"],
        index=pd.MultiIndex.from_frame(ann_datasets.test_multidx),
    )
    # group rows of dataframe based on hour and minute of day
    df_groups = df.groupby(
        [
            df.index.get_level_values("time").hour,
            df.index.get_level_values("time").minute,
        ]
    ).mean()
    df_groups.index.names = ["hour", "minute"]

    font_reg = FontProperties(weight="normal", size=8)
    font_bold = FontProperties(weight="bold", size=8)
    matplotlib.rcParams["font.size"] = 8
    matplotlib.rcParams["font.family"] = "sans-serif"

    margins = 20
    fig_width = 210 - margins * 2

    # make heating error plot for the whole year
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_width / 25.4, fig_width / 2 / 25.4)
    )
    fig.subplots_adjust(left=0.1)
    x_ticks = range(24)
    ax1.bar(x_ticks, df_groups["error heat [kWh]"], color="tomato")
    ax1.set_ylabel("Heating", fontproperties=font_bold, color="tomato")
    low, high, div, rnd_plcs = get_low_high_div_rnd(
        0, df_groups.max(axis=None), 7, fixed_low=True
    )
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    ax1.set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    ax1.set_ylim([low, high])
    x_ticklabels = [f"{row[0] + 1:02}" for row in df_groups.index]
    ax1.set_xticks(x_ticks, x_ticklabels)
    ax1.set_xticklabels(x_ticklabels, fontproperties=font_reg)
    ax1.set_xlim([-2 / 3, 23 + 2 / 3])
    for tic in ax1.xaxis.get_major_ticks():
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    for tic in ax1.xaxis.get_minor_ticks():
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    ax1.grid(which="both", color="k", linewidth=0.7, alpha=0.2)
    ax1.axvline(6, lw=1, c="k")
    ax1.text(
        5.5, high * 0.8, "weekday setpoint change", ha="right", fontproperties=font_reg
    )
    ax1.axvline(20, lw=1, c="k")
    ax1.text(
        19.5,
        high * 0.8,
        "weeknight setpoint change",
        ha="right",
        fontproperties=font_reg,
    )

    ax2.bar(x_ticks, df_groups["error cool [kWh]"], color="deepskyblue")
    ax2.set_ylabel("Cooling", fontproperties=font_bold, color="deepskyblue")
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    ax2.set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    ax1.set_ylim([low, high])
    ax2.set_xlabel("Hour of day", fontproperties=font_bold)
    ax2.set_xticks(x_ticks, x_ticklabels)
    ax2.set_xticklabels(x_ticklabels, fontproperties=font_reg)
    ax2.set_xlim([-2 / 3, 23 + 2 / 3])
    ax2.grid(which="both", color="k", linewidth=0.7, alpha=0.2)
    ax2.axvline(6, lw=1, c="k")
    ax2.axvline(20, lw=1, c="k")
    fig.text(
        0,
        0.54,
        "Mean absolute error [kWh]",
        fontproperties=font_bold,
        rotation=90,
        ha="left",
        va="center",
    )

    fig.savefig(
        Path(
            f"pred_elem_seq/datafiles/figures/error/test_lhs_{ann_datasets.cfg.n_lhs} - {ann_model_path.stem} - time of day error.pdf"
        ),
        bbox_inches="tight",
        pad_inches=0,
    )


def error_plots(ann_model_path, ann_datasets, run_ann=False):

    res = get_model_preds_prqt(
        ann_model_path, ann_datasets=ann_datasets, run_ann=run_ann
    )
    if ann_datasets.cfg.n_lhs_uni != 0:
        uni_perf = True
    else:
        uni_perf = False

    # res = {
    #     key: np.concat([res_arr[: 21 * 1000 * 8760], res_arr[22 * 1000 * 8760 :]])
    #     for key, res_arr in res.items()
    # }

    metrics = get_metrics(res, uni_perf)
    # instantiate dataframe of performance metrics
    res_metrics_df = pd.DataFrame(
        index=["Heating", "Cooling", "Average"],
        columns=["$MBE$", "$MAE$", "$MSE$", "$R^2$", "$R_{uni}^2$", r"$CVRMSE\ \%$"],
    )
    res_metrics_df.index.name = "Load type"
    res_metrics_df["$MBE$"] = np.append(metrics["mbe"], np.mean(metrics["mbe"]))
    res_metrics_df["$MAE$"] = np.append(metrics["mae"], np.mean(metrics["mae"]))
    res_metrics_df["$MSE$"] = np.append(metrics["mse"], np.mean(metrics["mse"]))
    res_metrics_df["$R^2$"] = np.append(metrics["r2"], np.mean(metrics["r2"]))
    res_metrics_df["$R_{uni}^2$"] = np.append(
        metrics["r2_uni"], np.mean(metrics["r2_uni"])
    )
    res_metrics_df[r"$CVRMSE\ \%$"] = np.append(
        metrics["cvrmse"], np.mean(metrics["cvrmse"])
    )

    if ann_datasets.cfg.n_lhs_uni != 0:
        start = f"test_lhs_uni_{ann_datasets.cfg.n_lhs_uni}"
    else:
        start = f"test_lhs_{ann_datasets.cfg.n_lhs}"
    res_metrics_df.to_csv(
        Path(
            f"pred_elem_seq/datafiles/results/{start} - {ann_model_path.stem} - error metrics.csv"
        )
    )

    # calculate error
    error = res["labels"] - res["preds"]
    # calculate absolute error
    error_abs = np.abs(error)
    # sort error arrays from smallest to largest
    error_heat = np.sort(error[:, 0])
    error_cool = np.sort(error[:, 1])
    error_heat_abs = np.flip(np.sort(error_abs[:, 0]))
    error_cool_abs = np.flip(np.sort(error_abs[:, 1]))
    # calculate mean absolute errors
    mae_heat = np.mean(error_heat_abs)
    mae_cool = np.mean(error_cool_abs)
    # calculate 1st and 3rd quartiles
    q5_heat = np.quantile(error_heat, 0.05)
    q95_heat = np.quantile(error_heat, 0.95)
    q5_cool = np.quantile(error_cool, 0.05)
    q95_cool = np.quantile(error_cool, 0.95)
    font_reg = FontProperties(weight="normal", size=8)
    font_bold = FontProperties(weight="bold", size=8)
    matplotlib.rcParams["font.size"] = 8
    matplotlib.rcParams["font.family"] = "sans-serif"
    # font_prop = {"family": my_font.get_name(), "size": 10 }
    # plt.rc("font", **font_prop)
    # plt.rc("axes", titlesize=8, titleweight="bold")
    margins = 20
    fig_width = 210 - margins * 2

    # make heating error plot for the whole year
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width / 25.4, 100 / 25.4))
    fig.subplots_adjust(left=0.1)
    ax1.plot(error_heat, lw=1, color="tomato")
    ax1.grid(which="both", axis="y", color="k", linewidth=0.7, alpha=0.2)
    ax1.axhline(lw=0.7, color="k")
    ax1.axhline(y=q5_heat, lw=0.7, color="tomato", linestyle=(0, (3, 3)))
    ax1.axhline(y=q95_heat, lw=0.7, color="tomato", linestyle=(0, (3, 3)))
    ax1.set_xlim(
        [0 - 20 * len(error_heat) / 8760, len(error_heat) + 20 * len(error_heat) / 8760]
    )
    low, high, div, rnd_plcs = get_low_high_div_rnd(min(error_heat), max(error_heat), 9)
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    ax1.set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    ax1.set_ylim([low, high])
    ax1.text(
        len(error_heat) / 2,
        q5_heat - 1.6 * (high - low) / 17.5,
        f"0.05 Quantile = {q5_heat:#.3g} kWh",
        ha="center",
        va="top",
        fontproperties=font_reg,
    )
    ax1.text(
        len(error_heat) / 2,
        q95_heat + 1.4 * (high - low) / 17.5,
        f"0.95 Quantile = {q95_heat:#.3g} kWh",
        ha="center",
        va="bottom",
        fontproperties=font_reg,
    )
    ax1.set_ylabel("Error [kWh]", fontproperties=font_bold)
    ax1.xaxis.get_major_formatter().set_scientific(False)
    for tic in ax1.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    for tic in ax1.xaxis.get_minor_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)

    ax2.plot(error_heat_abs, lw=1, color="tomato")
    ax2.grid(which="both", axis="y", color="k", linewidth=0.7, alpha=0.2)
    ax2.axhline(y=mae_heat, lw=0.7, color="tomato", linestyle=(0, (3, 3)))
    ax2.set_xlim(
        [
            0 - 20 * len(error_heat_abs) / 8760,
            len(error_heat_abs) + 20 * len(error_heat_abs) / 8760,
        ]
    )
    low, high, div, rnd_plcs = get_low_high_div_rnd(
        min(error_heat_abs), max(error_heat_abs), 7
    )
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    ax2.set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    ax2.set_ylim([low, high])
    ax2.text(
        8000 * len(error_heat_abs) / 8760,
        mae_heat + 0.65 * (high - low) / 10,
        f"Mean absolute error = {mae_heat:#.3g} kWh",
        ha="right",
        va="bottom",
        fontproperties=font_reg,
    )
    ax2.set_ylabel("Absolute error [kWh]", fontproperties=font_bold)
    ax2.xaxis.get_major_formatter().set_scientific(False)
    for tic in ax2.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    for tic in ax2.xaxis.get_minor_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    fig.text(
        0,
        0.525,
        "Heating",
        color="tomato",
        rotation=90,
        ha="left",
        va="center",
        fontproperties=font_bold,
    )

    fig.savefig(
        Path(
            f"pred_elem_seq/datafiles/figures/error/{start} - {ann_model_path.stem} - heat_err.pdf"
        ),
        bbox_inches="tight",
        pad_inches=0,
    )

    # make cooling error plot for the whole year
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width / 25.4, 100 / 25.4))
    fig.subplots_adjust(left=0.1)
    ax1.plot(error_cool, lw=1, color="deepskyblue")
    ax1.grid(which="both", axis="y", color="k", linewidth=0.7, alpha=0.2)
    ax1.axhline(lw=0.7, color="k")
    ax1.axhline(y=q5_cool, lw=0.7, color="deepskyblue", linestyle=(0, (3, 3)))
    ax1.axhline(y=q95_cool, lw=0.7, color="deepskyblue", linestyle=(0, (3, 3)))
    ax1.set_xlim(
        [0 - 20 * len(error_cool) / 8760, len(error_cool) + 20 * len(error_cool) / 8760]
    )
    low, high, div, rnd_plcs = get_low_high_div_rnd(
        min(error_cool), max(error_cool), 10
    )
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    ax1.set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    ax1.set_ylim([low, high])
    ax1.text(
        len(error_cool) / 2,
        q5_cool - 0.7 * (high - low) / 7,
        f"0.05 Quantile = {q5_cool:#.3g} kWh",
        ha="center",
        va="top",
        fontproperties=font_reg,
    )
    ax1.text(
        len(error_cool) / 2,
        q95_cool + 0.6 * (high - low) / 7,
        f"0.95 Quantile = {q95_cool:#.3g} kWh",
        ha="center",
        va="bottom",
        fontproperties=font_reg,
    )
    ax1.set_ylabel("Error [kWh]", fontproperties=font_bold)
    ax1.xaxis.get_major_formatter().set_scientific(False)
    for tic in ax1.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    for tic in ax1.xaxis.get_minor_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)

    ax2.plot(error_cool_abs, lw=1, color="deepskyblue")
    ax2.grid(which="both", axis="y", color="k", linewidth=0.7, alpha=0.2)
    ax2.axhline(y=mae_cool, lw=0.7, color="deepskyblue", linestyle=(0, (3, 3)))
    ax2.set_xlim(
        [
            0 - 20 * len(error_cool_abs) / 8760,
            len(error_cool_abs) + 20 * len(error_cool_abs) / 8760,
        ]
    )
    low, high, div, rnd_plcs = get_low_high_div_rnd(
        min(error_cool_abs), max(error_cool_abs), 6
    )
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    ax2.set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    ax2.set_ylim([low, high])
    ax2.text(
        8000 * len(error_cool_abs) / 8760,
        mae_cool + 0.25 * (high - low) / 3.5,
        f"Mean absolute error = {mae_cool:#.3g} kWh",
        ha="right",
        va="bottom",
        fontproperties=font_reg,
    )
    ax2.set_ylabel("Absolute error [kWh]", fontproperties=font_bold)
    ax2.xaxis.get_major_formatter().set_scientific(False)
    for tic in ax2.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    for tic in ax2.xaxis.get_minor_ticks():
        tic.tick1line.set_visible(False)
        tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    fig.text(
        0,
        0.525,
        "Cooling",
        color="deepskyblue",
        rotation=90,
        ha="left",
        va="center",
        fontproperties=font_bold,
    )

    fig.savefig(
        Path(
            f"pred_elem_seq/datafiles/figures/error/{start} - {ann_model_path.stem} - cool_err.pdf"
        ),
        bbox_inches="tight",
        pad_inches=0,
    )


# Custom function to remove leading zero
def custom_date_fmt(x, pos=None):
    dt = mdates.num2date(x)
    return dt.strftime("%b %d").replace("0", "")


def predict_year(ann_model_path, ann_datasets, run_ann):
    # get results for year
    res = get_model_preds_prqt(
        ann_model_path, ann_datasets=ann_datasets, run_ann=run_ann
    )
    # if the results array is 3D (because of pred_len > 1), index into first row and transpose so the hours of the year
    # are the rows
    if res["labels"].ndim == 3:
        res = {key: val[0].T for key, val in res.items()}

    bem_heat_y = res["labels"][:, 0]
    bem_cool_y = res["labels"][:, 1]
    # get orthogonal CNN prediction for univariate function
    surr_heat_y = res["preds"][:, 0]
    surr_cool_y = res["preds"][:, 1]

    metrics = get_metrics(res)
    # instantiate dataframe of performance metrics
    res_metrics_df = pd.DataFrame(
        index=["Heating", "Cooling", "Average"],
        columns=["$MBE$", "$MAE$", "$R^2$", r"$CVRMSE\ \%$"],
    )
    res_metrics_df.index.name = "Load type"
    res_metrics_df["$MBE$"] = np.append(metrics["mbe"], np.mean(metrics["mbe"]))
    res_metrics_df["$MAE$"] = np.append(metrics["mae"], np.mean(metrics["mae"]))
    res_metrics_df["$R^2$"] = np.append(metrics["r2"], np.mean(metrics["r2"]))
    res_metrics_df[r"$CVRMSE\ \%$"] = np.append(
        metrics["cvrmse"], np.mean(metrics["cvrmse"])
    )
    res_metrics_df.to_csv(
        Path(
            f"pred_elem_seq/datafiles/results/test_lhs_{ann_datasets.cfg.n_lhs} - {ann_model_path.stem} - error metrics - weather.csv"
        )
    )

    font_reg = FontProperties(weight="normal", size=8)
    font_bold = FontProperties(weight="bold", size=8)
    matplotlib.rcParams["font.size"] = 8
    matplotlib.rcParams["font.family"] = "sans-serif"

    building = ann_datasets.cfg.building
    x = building.sim_time_idx
    # get first week in February as datetimeindex and integer index
    x_winter = x[
        x.slice_indexer(f"{building.year}-02-01 00:30", f"{building.year}-02-07 23:30")
    ]
    winter_i = slice(x.get_loc(x_winter[0]), x.get_loc(x_winter[-1]) + 1)
    margins = 20
    fig_width = 210 - margins * 2
    # make heating plot of year
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_width / 25.4, 100 / 25.4), height_ratios=[3, 1]
    )
    ax1.plot(
        x_winter, bem_heat_y[winter_i], label="BEM", lw=2, zorder=0, color="tomato"
    )
    ax1.scatter(
        x_winter,
        surr_heat_y[winter_i],
        label="Surrogate model",
        s=1.7**2,
        color="k",
        zorder=10,
        # lw=2,
        # linestyle=(0, (1, 0.75)),
    )
    for tic in ax1.xaxis.get_major_ticks():
        # tic.tick1line.set_visible(False)
        # tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    for tic in ax1.xaxis.get_minor_ticks():
        # tic.tick1line.set_visible(False)
        # tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    low, high, div, rnd_plcs = get_low_high_div_rnd(
        0, max(bem_heat_y[winter_i]), 9, fixed_low=True
    )
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    ax1.set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    ax1.set_ylim([low, high])
    ax1.set_ylabel("Heating [kWh]", fontproperties=font_bold, color="tomato")
    ax1.legend(handlelength=1)
    for text in ax1.legend().get_texts():
        text.set_fontproperties(font_reg)
    ax1.grid(which="both", color="k", linewidth=0.7, alpha=0.2)
    # make absolute error heating plot
    error_heat = surr_heat_y - bem_heat_y
    ax2.axhline(lw=0.7, color="k")
    ax2.plot(x_winter, error_heat[winter_i], lw=1, color="tomato")
    low, high, div, rnd_plcs = get_low_high_div_rnd(
        min(error_heat[winter_i]), max(error_heat[winter_i]), 4
    )
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    ax2.set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    ax2.set_ylim([low, high])
    ax2.xaxis.set_major_formatter(FuncFormatter(custom_date_fmt))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.set_xticklabels(ax2.get_xticklabels(), fontproperties=font_reg)
    ax2.set_xlabel("Time", fontproperties=font_bold)
    ax2.set_ylabel("Heating Error [kWh]", fontproperties=font_bold, color="tomato")
    ax2.grid(which="both", color="k", linewidth=0.7, alpha=0.2)

    fig.savefig(
        Path(
            f"pred_elem_seq/datafiles/figures/year_pred/{ann_model_path.stem} - heat_feb - weather.pdf"
        ),
        bbox_inches="tight",
        pad_inches=0,
    )

    # get first week in August as datetimeindex and integer index
    x_summer = x[
        x.slice_indexer(f"{building.year}-08-02 00:30", f"{building.year}-08-08 23:30")
    ]
    summer_i = slice(x.get_loc(x_summer[0]), x.get_loc(x_summer[-1]) + 1)

    # make cooling plot of year
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_width / 25.4, 100 / 25.4), height_ratios=[3, 1]
    )
    ax1.plot(
        x_summer, bem_cool_y[summer_i], label="BEM", lw=2, zorder=0, color="deepskyblue"
    )
    ax1.scatter(
        x_summer,
        surr_cool_y[summer_i],
        label="Surrogate model",
        s=1.7**2,
        color="k",
        zorder=10,
        # lw=2,
        # linestyle=(0, (1, 0.75)),
    )
    for tic in ax1.xaxis.get_major_ticks():
        # tic.tick1line.set_visible(False)
        # tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    for tic in ax1.xaxis.get_minor_ticks():
        # tic.tick1line.set_visible(False)
        # tic.tick2line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    low, high, div, rnd_plcs = get_low_high_div_rnd(
        0, max(bem_cool_y[summer_i]), 9, fixed_low=True
    )
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    ax1.set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    ax1.set_ylim([low, high])
    ax1.set_ylabel("Cooling [kWh]", fontproperties=font_bold, color="deepskyblue")
    ax1.legend(handlelength=1)
    for text in ax1.legend().get_texts():
        text.set_fontproperties(font_reg)
    ax1.grid(which="both", color="k", linewidth=0.7, alpha=0.2)
    # make absolute error heating plot
    error_cool = surr_cool_y - bem_cool_y
    ax2.axhline(lw=0.7, color="k")
    ax2.plot(x_summer, error_cool[summer_i], lw=1, color="deepskyblue")
    low, high, div, rnd_plcs = get_low_high_div_rnd(
        min(error_cool[summer_i]), max(error_cool[summer_i]), 5
    )
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    ax2.set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    ax2.set_ylim([low, high])
    ax2.xaxis.set_major_formatter(FuncFormatter(custom_date_fmt))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.set_xticklabels(ax2.get_xticklabels(), fontproperties=font_reg)
    ax2.set_xlabel("Time", fontproperties=font_bold)
    ax2.set_ylabel("Cooling Error [kWh]", fontproperties=font_bold, color="deepskyblue")
    ax2.grid(which="both", color="k", linewidth=0.7, alpha=0.2)

    fig.savefig(
        Path(
            f"pred_elem_seq/datafiles/figures/year_pred/{ann_model_path.stem} - cool_aug - weather.pdf"
        ),
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_climate_map_errors(ann_model_path, ann_datasets, weather_list):
    ann_model_res = get_model_preds_prqt(ann_model_path, ann_datasets, run_ann=False)
    # reshape outputs into 3D array so that each weather station prediction is separated
    res_labels = ann_model_res["labels"].reshape(30, 1000 * 8760, -1)
    res_preds = ann_model_res["preds"].reshape(30, 1000 * 8760, -1)

    # create pandas Dataframe to hold R^2 and MAE results
    res_per_weather = pd.DataFrame(
        index=weather_list,
        columns=["R2 heat", "R2 cool", "R2 mean", "MAE heat", "MAE cool"],
    )
    # iterate through each row of the reshaped outputs (i.e., each weather file)
    for i, (labels, preds) in enumerate(zip(res_labels, res_preds)):
        # recombine reshaped results into a dictionary
        res = {"labels": labels, "preds": preds}
        # get R^2 and MAE metrics
        metrics = get_metrics(res)
        # assign metrics to dataframe
        res_per_weather.iloc[i, :] = np.concat(
            [metrics["r2"], [np.mean(metrics["r2"])], metrics["mae"]]
        )

    # sort dataframe from best to worst mean R^2
    res_per_weather = res_per_weather.sort_values(by="R2 mean", ascending=False)
    # write results to file
    res_per_weather.to_csv(
        Path("pred_elem_seq/datafiles/figures/climate_map/results_per_weather.csv")
    )

    font_reg = FontProperties(weight="normal", size=8)
    font_small = FontProperties(weight="normal", size=6)
    font_bold = FontProperties(weight="bold", size=8)
    font_bold_large = FontProperties(weight="bold", size=12)
    matplotlib.rcParams["font.size"] = 8
    matplotlib.rcParams["font.family"] = "sans-serif"

    # instantiate figure with dimensions
    margins = 20
    fig_width = 210 - margins * 2
    fig_height = 100
    fig, axes = plt.subplots(
        nrows=4, ncols=1, sharex=True, figsize=(fig_width / 25.4, fig_height / 25.4)
    )
    fig.subplots_adjust(left=0.1, top=0.94)

    axes[0].scatter(
        range(res_per_weather.shape[0]), res_per_weather["R2 heat"], color="tomato"
    )
    low, high, div, rnd_plcs = get_low_high_div_rnd(
        res_per_weather["R2 heat"].min(), 1.0, 4, fixed_high=True
    )
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    axes[0].set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    axes[0].set_ylim([low, high])
    axes[0].set_ylabel("R²", fontproperties=font_reg)
    axes[0].grid(which="major", axis="x", color="k", linewidth=0.7, alpha=0.2)
    for i in range(res_per_weather.shape[0]):
        axes[0].text(i, 1.05, i + 1, ha="center", va="top", fontproperties=font_bold)

    axes[1].scatter(
        range(res_per_weather.shape[0]), res_per_weather["MAE heat"], color="tomato"
    )
    low, high, div, rnd_plcs = get_low_high_div_rnd(
        res_per_weather["MAE heat"].min(), res_per_weather["MAE heat"].max(), 5
    )
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    axes[1].set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    axes[1].set_ylim([low, high])
    axes[1].set_ylabel("MAE [kWh]", fontproperties=font_reg)
    axes[1].grid(which="major", axis="x", color="k", linewidth=0.7, alpha=0.2)
    axes[1].grid(which="major", axis="x", color="k", linewidth=0.7, alpha=0.2)
    fig.text(
        0,
        0.74,
        "Heating",
        color="tomato",
        rotation=90,
        ha="left",
        va="center",
        fontproperties=font_bold,
    )

    axes[2].scatter(
        range(res_per_weather.shape[0]), res_per_weather["R2 cool"], color="deepskyblue"
    )
    low, high, div, rnd_plcs = get_low_high_div_rnd(
        res_per_weather["R2 cool"].min(), 1.0, 5, fixed_high=True
    )
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    axes[2].set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    axes[2].set_ylim([low, high])
    axes[2].set_ylabel("R²", fontproperties=font_reg)
    axes[2].grid(which="major", axis="x", color="k", linewidth=0.7, alpha=0.2)

    axes[3].scatter(
        range(res_per_weather.shape[0]),
        res_per_weather["MAE cool"],
        color="deepskyblue",
    )
    low, high, div, rnd_plcs = get_low_high_div_rnd(
        res_per_weather["MAE cool"].min(),
        res_per_weather["MAE cool"].max(),
        5,
    )
    yticks = np.arange(low, high + div, div).round(rnd_plcs)
    axes[3].set_yticks(yticks, [str(y) for y in yticks], fontproperties=font_reg)
    axes[3].set_ylim([low, high])
    axes[3].set_ylabel("MAE [kWh]", fontproperties=font_reg)
    axes[3].tick_params(axis="x", which="major", pad=16)
    axes[3].grid(which="major", axis="x", color="k", linewidth=0.7, alpha=0.2)
    axes[3].set_xlim([-0.5, len(res_per_weather.index) - 0.5])
    axes[3].set_xticks(
        range(res_per_weather.shape[0]),
        [
            re.search(r".*(?=\.\d{6})", idx.stem).group(0)
            for idx in res_per_weather.index
        ],
        rotation=90,
        fontproperties=font_small,
    )
    fig.text(
        0,
        0.31,
        "Cooling",
        color="deepskyblue",
        rotation=90,
        ha="left",
        va="center",
        fontproperties=font_bold,
    )
    for i in range(res_per_weather.shape[0]):
        axes[3].text(i, 0.9, i + 1, ha="center", va="top", fontproperties=font_bold)

    fig.text(
        0.02,
        0.96,
        "(b)",
        ha="center",
        va="center",
        fontproperties=font_bold_large,
    )

    fig.savefig(
        "pred_elem_seq/datafiles/figures/climate_map/Climate map errors.pdf",
        bbox_inches="tight",
        pad_inches=0,
    )


def make_climate_map_paul():
    list_epws = list(Path("pred_elem_seq/datafiles/weather/ninja/train").iterdir())
    data = pd.DataFrame(
        {
            "lat": [
                pvlib.iotools.read_epw(epw_path)[1]["latitude"]
                for epw_path in list_epws
            ],
            "lon": [
                pvlib.iotools.read_epw(epw_path)[1]["longitude"]
                for epw_path in list_epws
            ],
            "name": [
                pvlib.iotools.read_epw(epw_path)[1]["city"] for epw_path in list_epws
            ],
        }
    )
    list_epws_unseen = pd.read_csv(
        Path("pred_elem_seq/datafiles/figures/climate_map/results_per_weather.csv"),
        index_col=0,
    ).index
    data_unseen = pd.DataFrame(
        {
            "lat": [
                pvlib.iotools.read_epw(epw_unseen)[1]["latitude"]
                for epw_unseen in list_epws_unseen
            ],
            "lon": [
                pvlib.iotools.read_epw(epw_unseen)[1]["longitude"]
                for epw_unseen in list_epws_unseen
            ],
            "name": [
                pvlib.iotools.read_epw(epw_unseen)[1]["city"]
                for epw_unseen in list_epws_unseen
            ],
        }
    )
    src = rasterio.open(
        Path(
            "pred_elem_seq/datafiles/figures/climate_map/Beck_KG_V1_present_0p083.tiff"
        )
    )

    font_reg = FontProperties(weight="normal", size=19)
    font_numb = FontProperties(weight="normal", size=25)
    font_bold = FontProperties(weight="bold", size=32)
    matplotlib.rcParams["font.size"] = 8
    matplotlib.rcParams["font.family"] = "sans-serif"
    
    fig = plt.figure(figsize=(20, 10))

    cmap = [
        "#08306b",
        "#08519c",
        "#2171b5",
        "#4292c6",
        "#6baed6",
        "#9ecae1",
        "#c6dbef",
        "#deebf7",
        "#fee0d2",
        "#fcbba1",
        "#fc9272",
        "#fb6a4a",
        "#ef3b2c",
        "#cb181d",
        "#a50f15",
        "#67000d",
        "#FFFFFF",
    ]
    cmap = [cmap[i] for i in range(len(cmap) - 1, 0, -1)]
    cmap = ListedColormap(cmap)
    neg = plt.imshow(src.read(1), cmap="Blues")
    for i in range(0, len(list_epws)):
        if i == 0:
            plt.scatter(
                (data.iloc[i]["lon"] + 180) * 12,
                (data.iloc[i]["lat"] * -1 + 90) * 12,
                color="mediumseagreen",
                edgecolors="black",
                s=40,
                label="Training weather stations",
                zorder=10,
            )
        else:
            plt.scatter(
                (data.iloc[i]["lon"] + 180) * 12,
                (data.iloc[i]["lat"] * -1 + 90) * 12,
                color="mediumseagreen",
                edgecolors="black",
                s=40,
                zorder=10,
            )
        # plt.text((data.iloc[i]['lon']+180)*12,(data.iloc[i]['lat']*-1+90)*12,str(i), fontsize=19, color='green')

    for i in range(0, len(list_epws_unseen)):
        if i == 0:
            plt.scatter(
                (data_unseen.iloc[i]["lon"] + 180) * 12,
                (data_unseen.iloc[i]["lat"] * -1 + 90) * 12,
                color="orangered",
                edgecolors="black",
                marker="d",
                s=40,
                label="Testing weather stations",
                zorder=10,
            )
        else:
            plt.scatter(
                (data_unseen.iloc[i]["lon"] + 180) * 12,
                (data_unseen.iloc[i]["lat"] * -1 + 90) * 12,
                color="orangered",
                edgecolors="black",
                marker="d",
                s=40,
                zorder=10,
            )
        plt.text(
            (data_unseen.iloc[i]["lon"] + 180) * 12 - 15,
            (data_unseen.iloc[i]["lat"] * -1 + 90) * 12,
            str(i + 1),
            color="orangered",
            horizontalalignment="right",
            fontproperties=font_numb,
        )
    fig.text(
        0.145,
        0.85,
        "(a)",
        ha="center",
        va="center",
        fontproperties=font_bold,
    )
    ax = plt.gca()
    _ = ax.set_xticks([])
    _ = ax.set_yticks([])
    plt.legend()
    for text in plt.legend().get_texts():
        text.set_fontproperties(font_reg)
    plt.savefig(
        Path("pred_elem_seq/datafiles/figures/climate_map/Climate map.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )
