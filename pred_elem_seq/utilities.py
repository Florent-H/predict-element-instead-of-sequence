import json
import os
import pickle
import pandas as pd
import numpy as np


def write_dic_to_file(dictionary, dictionary_path, encoding=None, ensure_ascii=True):
    # print dictionary (could be problem or results)
    """
    Args:
        dictionary:
        dictionary_path:
        encoding:
        ensure_ascii:
    """
    if isinstance(encoding, str):
        with open(dictionary_path, "w", encoding=encoding) as fw:
            json.dump(dictionary, fw, ensure_ascii=ensure_ascii, indent=4)
            fw.flush()
            os.fsync(fw)
    else:
        with open(dictionary_path, "w") as fw:
            json.dump(dictionary, fw, ensure_ascii=ensure_ascii, indent=4)
            fw.flush()
            os.fsync(fw)


def read_dic_from_file(dictionary_path, encoding=None):
    if isinstance(encoding, str):
        with open(dictionary_path, "r", encoding=encoding) as fr:
            dictionary = json.load(fr)
    else:
        with open(dictionary_path, "r") as fr:
            dictionary = json.load(fr)

    return dictionary


def numb_param_style(x, n_sig_figs=3):
    if x == 0:
        return "0"
    # Calculate magnitude
    mag = int(np.floor(np.log10(abs(x))))
    # Calculate required decimals for 3 sig figs
    decimals = max(0, n_sig_figs - 1 - mag)
    return f"{x:,.{decimals}f}M"


def df_to_latex(df_path, vertical_header=False, hide_index=True, unit="kWh"):
    df = pd.read_csv(df_path, index_col=False)

    # Create a Styler where the last column (i.e., "Train time") has three significant figures
    styler = df.style.format(
        {
            "Number of parameters": lambda x: numb_param_style(x / 1e6),
            "Train time [h]": lambda x: f"{x:#.3g}",
            "Weight decay": lambda x: f"{x:.1}",
            "Starting learning rate": lambda x: f"{x:.1}",
            "Max learning rate": lambda x: f"{x:.1}",
            "$MAE$": lambda x: f"{x:#.6g}",
            "${MAE}_{heat}$": lambda x: f"{x:#.6g}",
            "${MAE}_{cool}$": lambda x: f"{x:#.6g}",
            "${MAE}_{avg}$": lambda x: f"{x:#.6g}",
            "$MBE$": lambda x: f"{x:#.6g}",
            "${MBE}_{heat}$": lambda x: f"{x:#.6g}",
            "${MBE}_{cool}$": lambda x: f"{x:#.6g}",
            "${MBE}_{avg}$": lambda x: f"{x:#.6g}",
            r"$MAPE\ \%$": lambda x: f"{x:#.6g}",
            r"${MAPE}_{heat}\ \%$": lambda x: f"{x:#.6g}",
            r"${MAPE}_{cool}\ \%$": lambda x: f"{x:#.6g}",
            r"${MAPE}_{avg}\ \%$": lambda x: f"{x:#.6g}",
            "$R^2$": lambda x: f"{x:#.6g}",
            "$R_{heat}^2$": lambda x: f"{x:#.6g}",
            "$R_{cool}^2$": lambda x: f"{x:#.6g}",
            "$R_{avg}^2$": lambda x: f"{x:#.6g}",
            "$R_{uni}^2$": lambda x: f"{x:#.6g}",
            "$R_{uni,heat}^2$": lambda x: f"{x:#.6g}",
            "$R_{uni,cool}^2$": lambda x: f"{x:#.6g}",
            "$R_{uni,avg}^2$": lambda x: f"{x:#.6g}",
            r"$CVRMSE\ \%$": lambda x: f"{x:#.6g}",
            r"${CVRMSE}_{heat}\ \%$": lambda x: f"{x:#.6g}",
            r"${CVRMSE}_{cool}\ \%$": lambda x: f"{x:#.6g}",
            r"${CVRMSE}_{avg}\ \%$": lambda x: f"{x:#.6g}",
        }
    )
    # generate string
    if hide_index:
        latex_str = styler.hide(axis="index").to_latex(hrules=True)
    else:
        latex_str = styler.to_latex(hrules=True)

    # get original headers
    headers = list(df.columns)
    for header in headers:
        # if percentage symbol is at the end of the header
        if any(metric in header for metric in ["MBE", "MAE"]):
            end = f"\ [{unit}]"
        else:
            end = ""

        # if the header text is to be rotated by 90 degrees
        if vertical_header:
            # if header contains $$ (i.e., math):
            if "$" in header:
                # replace math header with $\mathbf{...}$
                latex_str = latex_str.replace(
                    header,
                    f"\\rotatebox{{90}}{{$\\mathbf{{{header.replace("$", "")}{end}}}$}}",
                )
            else:
                # replace other headers with \textbf{...}
                latex_str = latex_str.replace(
                    header, f"\\rotatebox{{90}}{{\\textbf{{{header}}}}}"
                )
        else:
            # if header contains $$ (i.e., math):
            if "$" in header:
                # replace math header with $\mathbf{...}$
                # get header without percentage at the end
                latex_str = latex_str.replace(
                    header,
                    f"$\\mathbf{{{header.replace("$", "")}{end}}}$",
                )
            else:
                # replace other headers with \textbf{...}
                latex_str = latex_str.replace(header, f"\\textbf{{{header}}}")
    # Replace tabular environment with tabularx
    latex_str = latex_str.replace(r"\begin{tabular}", r"\begin{tabularx}{\linewidth}")
    latex_str = latex_str.replace(r"\end{tabular}", r"\end{tabularx}")
    # get path of latex string
    latex_str_path = df_path.parent / f"{df_path.stem}-latex.txt"
    # write to file
    with open(latex_str_path, "w") as fw:
        fw.write(latex_str)


def get_n_model_params(ann_model_path):
    # load torch model from disk
    with open(ann_model_path, "rb") as f:
        ann_model = pickle.load(f)

    pytorch_total_params = sum(p.numel() for p in ann_model.parameters())
    print(pytorch_total_params)
    return pytorch_total_params
