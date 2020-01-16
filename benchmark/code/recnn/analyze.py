# This file is part of the Reproducible Open Benchmarks for Data Analysis
# Platform (ROB) - Top Tagger Benchmark Demo.
#
# Copyright (C) [2019-2020] NYU.
#
# ROB is free software; you can redistribute it and/or modify it under the
# terms of the MIT License; see LICENSE file for more details.

import numpy as np

from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


def get_median_bg_reject(
    tagger_result, true_labels, tag_eff=0.5, remove_outliers=True
):
    """Get median model sorted by auc"""
    # Get tag efficiency working point
    if tag_eff == 0.5:
        point = 225
    elif tag_eff == 0.3:
        point = 125
    elif tag_eff == 0.4:
        point = 175
    # Initialize local variables
    base_tpr = np.linspace(0.05, 1, 476)
    all_bg_reject = []
    all_bg_reject_std = []
    all_bg_reject_outliers = []
    all_bg_reject_std_outliers = []
    all_aucs = []
    all_auc_std = []
    all_aucs_outliers = []
    all_auc_std_outliers = []
    len_models = []

    bg_reject = []
    aucs = []
    print('Tagger has {} runs'.format(len(tagger_result)))
    for i in range(len(tagger_result)):
        auc = roc_auc_score(true_labels, tagger_result[i])
        aucs.append(auc)
        fpr, tpr, thresholds = roc_curve(
            true_labels,
            tagger_result[i],
            pos_label=1,
            drop_intermediate=False
        )
        interp_fpr = interp(base_tpr, tpr, fpr)
        bg_reject.append(1./interp_fpr[point])
    all_bg_reject.append(np.median(bg_reject))
    all_bg_reject_std.append(np.std(bg_reject))
    all_aucs.append(np.median(aucs))
    all_auc_std.append(np.std(aucs))
    if remove_outliers:
        scores = np.asarray(bg_reject)
        p25 = np.percentile(scores, 1 / 6. * 100.)
        p75 = np.percentile(scores, 5 / 6. * 100)
        # Get mean and std for the bg rejection
        robust_mean = np.mean([scores[i] for i in range(len(scores)) if p25 <= scores[i] <= p75])
        robust_std = np.std([scores[i] for i in range(len(scores)) if p25 <= scores[i] <= p75])
        indices = [i for i in range(len(scores)) if robust_mean - 3*robust_std <= scores[i] <= robust_mean + 3*robust_std]
        new_scores=scores[indices]
        len_models.append(len(new_scores))
        all_bg_reject_outliers.append(np.median(new_scores))
        all_bg_reject_std_outliers.append(np.std(new_scores))
        new_aucs = np.asarray(aucs)[indices]
        all_aucs_outliers.append(np.median(new_aucs))
        all_auc_std_outliers.append(np.std(new_aucs))
    if remove_outliers:
        return (
            all_bg_reject_outliers,
            all_bg_reject_std_outliers,
            all_aucs_outliers,
            all_auc_std_outliers
        )
    else:
        return (all_bg_reject, all_bg_reject_std, all_aucs, all_auc_std)
