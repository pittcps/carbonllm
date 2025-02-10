import pandas as pd
import numpy as np
import ast
import re

def is_dict_list(lst):
    return all(isinstance(i, dict) for i in lst)

def convert_dict_list_to_float_list(dict_list):
    float_list = []
    for d in dict_list:
        key, value = next(iter(d.items()))
        try:
            float_list.append(float(value))
        except ValueError:
            float_list.append(0.0)
    return float_list

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(item)
        else:
            flat_list.append(item)
    return flat_list

def pad_or_truncate(pred, length):
    pred = [float(str(x)) if isinstance(x, (int, float)) or str(x).replace('.', '', 1).isdigit() else 0.0 for x in pred]
    if len(pred) < length:
        pred = pred + [0] * (length - len(pred))
    return np.array(pred[:length], dtype=float)

def parse_evidence(evidence):
    if not evidence:
        return [], []
    try:
        evidence_dict = ast.literal_eval(evidence)
    except (SyntaxError, ValueError):
        evidence_dict = {}
    values = list(evidence_dict.values())
    keys = [list(map(int, re.findall(r'\d+', k))) for k in evidence_dict.keys()]
    return values, keys

def compute_metrics(y_true, y_pred, token_true, token_pred):
    rmse_list = []
    mae_list = []
    em_list = []
    token_em_list = []
    total_cases = 0
    gt_zero_count = 0
    pred_zero_count = 0

    for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
        if is_dict_list(true):
            true = convert_dict_list_to_float_list(true)
        if is_dict_list(pred):
            pred = convert_dict_list_to_float_list(pred)

        true = flatten_list(true)
        pred = flatten_list(pred)

        if not true and not pred:
            true = [1]
            pred = [1]

        if not true:
            true = [0] * len(pred)
        elif not pred:
            pred = [0] * len(true)

        true = np.array(true, dtype=float)
        pred = pad_or_truncate(pred, len(true))

        if np.all(true == 0):
            gt_zero_count += 1
        if np.all(pred == 0):
            pred_zero_count += 1

        rmse = np.mean((true - pred) ** 2)
        rmse = rmse ** 0.5
        mae = np.mean(np.abs(true - pred))
        if rmse <= 40000:
            rmse_list.append(rmse)
            mae_list.append(mae)
        else:
            total_diff = np.sum(np.abs(true - pred))
            print(f"Row {idx + 2}: GT: {true.tolist()} Predict: {pred.tolist()} Total difference: {total_diff:.4f} RMSE: {rmse:.4f} MAE: {mae:.4f}")
            rmse_list.append(0)
            mae_list.append(0)

        exact_match = np.allclose(true, pred, atol=0.01)
        em_list.append(1 if exact_match else 0)

        token_exact_match = token_true[idx] == token_pred[idx]
        token_em_list.append(1 if token_exact_match else 0)

        total_cases += 1

    avg_rmse = sum(rmse_list)/len(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_em = np.mean(em_list) * 100
    avg_token_em = np.mean(token_em_list) * 100

    gt_zero_percent = (gt_zero_count / total_cases) * 100
    pred_zero_percent = (pred_zero_count / total_cases) * 100

    return avg_rmse, avg_mae, avg_em, avg_token_em, gt_zero_percent, pred_zero_percent

gt_df = pd.read_csv('../output/test.csv')
predict_df = pd.read_csv('../input/prediction.csv')

gt_list = gt_df['Ground truth answer'].apply(
    lambda x: [] if x == '-' else ast.literal_eval(x)
).tolist()
predict_list = []
for item in predict_df['Answer']:
    try:
        value = ast.literal_eval(item)
        if isinstance(value, list):
            predict_list.append(value)
        else:
            predict_list.append([])
    except (ValueError, SyntaxError):
        predict_list.append([])

token_true = gt_df['Relevance token'].tolist()
token_pred = predict_df['Relevance token'].tolist()

min_length = min(len(gt_list), len(predict_list))
gt_list = gt_list[:min_length]
predict_list = predict_list[:min_length]
token_true = token_true[:min_length]
token_pred = token_pred[:min_length]

avg_rmse, avg_mae, avg_em, avg_token_em, gt_zero_percent, pred_zero_percent = compute_metrics(gt_list, predict_list, token_true, token_pred)

print("Total samples:", min_length)
print(f"RMSE: {avg_rmse:.2f}")
print(f"MAE: {avg_mae:.2f}")
print(f"EM: {avg_em:.2f}%")
print(f"Relevance Token EM: {avg_token_em:.2f}%")
