import numpy as np
import torch
import torch.nn as nn



class MarginalSelection:
    def __init__(self, alpha, random_state=2023):
        self.alpha = alpha
        self.random_state = random_state

    def multiclass_classification(self, X_tests, X_calib, Y_calib, backbone, left_tail=False, conditional=True, device='cpu'):
        labels = np.array(sorted(set(Y_calib.tolist() if hasattr(Y_calib, 'tolist') else list(Y_calib))))
        cal_scores = nonconf_scores_mc_np(X_calib, Y_calib, backbone, device=device)
        n_test = X_tests.shape[0]
        conf_pval = np.full((n_test, len(labels)), -np.inf)
        for idx, y in enumerate(labels):
            y_test = np.repeat(y, n_test)
            scores_test_y = nonconf_scores_mc_np(X_tests, y_test, backbone, device=device)
            idx_y = np.where(Y_calib == y)[0] if conditional else np.arange(len(Y_calib))
            cal_scores_y = cal_scores[idx_y]
            for i in range(n_test):
                conf_pval[i, idx] = compute_conf_pvals_np(scores_test_y[i], cal_scores_y, left_tail=left_tail)
        C_full = []
        for i in range(n_test):
            C_set = np.where(conf_pval[i] >= self.alpha)[0].tolist()
            C_full.append(C_set)
        return C_full


class PartialSelection:
    def __init__(self, alpha, random_state=2023):
        self.alpha = alpha
        self.random_state = random_state

    def multiclass_classification(self, X_tests, X_calib, Y_calib, backbone, left_tail=False, sensitive_atts_idx=None, conditional=True, device='cpu'):
        labels = np.array(sorted(set(Y_calib.tolist() if hasattr(Y_calib, 'tolist') else list(Y_calib))))
        cal_scores = nonconf_scores_mc_np(X_calib, Y_calib, backbone, device=device)
        n_test = X_tests.shape[0]
        conf_pval = np.full((n_test, len(labels)), -np.inf)
        for idx, y in enumerate(labels):
            y_test = np.repeat(y, n_test)
            scores_test_y = nonconf_scores_mc_np(X_tests, y_test, backbone, device=device)
            idx_y = np.where(Y_calib == y)[0] if conditional else np.arange(len(Y_calib))
            X_calib_y = X_calib[idx_y]
            cal_scores_y = cal_scores[idx_y]
            for i, X_test in enumerate(X_tests):
                vals = []
                for att_idx in sensitive_atts_idx:
                    mask = np.where(X_calib_y[:, att_idx] == X_test[att_idx])[0]
                    vals.append(compute_conf_pvals_np(scores_test_y[i], cal_scores_y[mask], left_tail=left_tail))
                conf_pval[i, idx] = np.max(vals) if vals else compute_conf_pvals_np(scores_test_y[i], cal_scores_y, left_tail=left_tail)
        C_full = []
        for i in range(n_test):
            C_set = np.where(conf_pval[i] >= self.alpha)[0].tolist()
            C_full.append(C_set)
        return C_full


class ExhaustiveSelection:
    def __init__(self, alpha, random_state=2023):
        self.alpha = alpha
        self.random_state = random_state

    def multiclass_classification(self, X_tests, X_calib, Y_calib, backbone, left_tail=False, sensitive_atts_idx=None, conditional=True, device='cpu'):
        labels = np.array(sorted(set(Y_calib.tolist() if hasattr(Y_calib, 'tolist') else list(Y_calib))))
        cal_scores = nonconf_scores_mc_np(X_calib, Y_calib, backbone, device=device)
        n_test = X_tests.shape[0]
        conf_pval = np.full((n_test, len(labels)), -np.inf)
        for idx, y in enumerate(labels):
            y_test = np.repeat(y, n_test)
            scores_test_y = nonconf_scores_mc_np(X_tests, y_test, backbone, device=device)
            idx_y = np.where(Y_calib == y)[0] if conditional else np.arange(len(Y_calib))
            X_calib_y = X_calib[idx_y]
            cal_scores_y = cal_scores[idx_y]
            for i, X_test in enumerate(X_tests):
                mask = np.all(X_calib_y[:, sensitive_atts_idx] == X_test[sensitive_atts_idx], axis=1)
                selected = cal_scores_y[mask]
                conf_pval[i, idx] = compute_conf_pvals_np(scores_test_y[i], selected, left_tail=left_tail)
        C_full = []
        for i in range(n_test):
            C_set = np.where(conf_pval[i] >= self.alpha)[0].tolist()
            C_full.append(C_set)
        return C_full



# =========================
# AFCP
# =========================


def backbone_probs_numpy(backbone, X_np, batch_size=512, device='cpu'):
    backbone.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.tensor(X_np[i:i+batch_size], dtype=torch.float32, device=device)
            logits, _ = backbone(xb)
            probs = torch.softmax(logits, dim=-1)
            outs.append(probs.cpu().numpy())
    return np.concatenate(outs, axis=0)


def nonconf_scores_mc_np(X_np, Y_np, backbone, device='cpu'):
    probs = backbone_probs_numpy(backbone, X_np, device=device)
    idx = np.arange(len(Y_np))
    return 1.0 - probs[idx, Y_np.astype(int)]


def compute_conf_pvals_np(test_score, calib_scores, left_tail=False):
    calib_scores = np.asarray(calib_scores, dtype=float)
    if calib_scores.size == 0:
        return 1.0
    if left_tail:
        return (1.0 + np.sum(calib_scores <= test_score)) / (len(calib_scores) + 1.0)
    return (1.0 + np.sum(calib_scores >= test_score)) / (len(calib_scores) + 1.0)


def arc_wrapper_np(calib_scores, x_single, backbone, alpha, device='cpu'):
    probs = backbone_probs_numpy(backbone, x_single, device=device)
    scores = 1.0 - probs[0]
    pred_set = []
    for y, s in enumerate(scores):
        pval = compute_conf_pvals_np(s, calib_scores, left_tail=False)
        if pval >= alpha:
            pred_set.append(y)
    return [pred_set]


class AFCPAdaptiveSelection:
    def __init__(self, alpha, ttest_delta=None, random_state=2024):
        self.alpha = alpha
        self.random_state = random_state
        self.beta = 0.0
        self.ttest_delta = ttest_delta
        self.sig_level = 0.1

    def augment_data(self, X_calib, Y_calib, x_test, y):
        X = np.vstack([X_calib, x_test[None, :]])
        Y = np.append(Y_calib, int(y))
        return X, Y

    def error_func_groupwise(self, phi_k, E):
        max_miscov = -1.0
        miscov_ind = np.array(E, dtype=float)
        phi_k = np.asarray(phi_k)
        E = np.asarray(E, dtype=float)
        for m in np.unique(phi_k):
            grouped = phi_k == m
            frac = grouped.mean()
            if frac < self.beta or grouped.sum() == 0:
                continue
            miscov_prop = E[grouped].mean()
            if miscov_prop >= max_miscov:
                max_miscov = float(miscov_prop)
                miscov_ind = E[grouped]
        if max_miscov < 0:
            max_miscov = float(E.mean()) if len(E) else 0.0
            miscov_ind = E
        return max_miscov, np.asarray(miscov_ind, dtype=float)

    def select_the_worst_group(self, att_idx, E, X_aug):
        max_max_miscov = -1.0
        best_att = []
        best_miscov_ind = np.asarray(E, dtype=float)
        for att in att_idx:
            max_cov_temp, miscov_ind_temp = self.error_func_groupwise(phi_k=X_aug[:, att], E=E)
            if max_cov_temp >= max_max_miscov:
                max_max_miscov = max_cov_temp
                best_att = att
                best_miscov_ind = miscov_ind_temp
        if self.ttest_delta is not None and len(best_miscov_ind) > 1:
            test_result = stats.ttest_1samp(best_miscov_ind, self.alpha + self.ttest_delta, alternative='greater')
            if not np.isfinite(test_result.pvalue) or test_result.pvalue >= self.sig_level:
                return []
        return best_att

    def multiclass_classification(self, X_calib, Y_calib, X_test, backbone, att_idx, return_khat=True, conditional=False, left_tail=False, device='cpu'):
        n_test = X_test.shape[0]
        labels = np.array(sorted(set(Y_calib.tolist() if hasattr(Y_calib, 'tolist') else list(Y_calib))))
        k_final = []
        C_sets_final = []
        calib_scores = nonconf_scores_mc_np(X_calib, Y_calib, backbone, device=device)
        test_scores = np.full((n_test, len(labels)), -np.inf)
        conf_pval_y = np.full((n_test, len(labels)), -np.inf)
        conf_pval_add = np.full((n_test, len(labels)), -np.inf)
        for idx, y in enumerate(labels):
            test_scores[:, idx] = nonconf_scores_mc_np(X_test, np.repeat(y, n_test), backbone, device=device)
        for i, x in enumerate(X_test):
            k_hat_i = []
            for idx, y in enumerate(labels):
                idx_y = np.where(Y_calib == y)[0] if conditional else np.arange(len(Y_calib))
                X_calib_y = X_calib[idx_y]
                Y_calib_y = Y_calib[idx_y]
                X_aug, Y_aug = self.augment_data(X_calib_y, Y_calib_y, x, y)
                scores_aug = np.append(calib_scores[idx_y], test_scores[i, idx])
                n_union = X_aug.shape[0]
                E = []
                for j in range(n_union):
                    not_j = np.array([k for k in range(n_union) if k != j], dtype=int)
                    C_temp = arc_wrapper_np(scores_aug[not_j], X_aug[j][None, :], backbone, self.alpha, device=device)
                    E.append(float(int(Y_aug[j]) not in C_temp[0]))
                k_hat = self.select_the_worst_group(att_idx, E, X_aug)
                if k_hat == []:
                    k_hat_i.append(set())
                    calib_scores_selected = scores_aug[:-1]
                else:
                    k_hat_i.append({int(k_hat)})
                    mask = np.where(X_calib_y[:, k_hat] == x[k_hat])[0]
                    calib_scores_selected = np.asarray(scores_aug[:-1])[mask]
                conf_pval_y[i, idx] = compute_conf_pvals_np(scores_aug[-1], calib_scores_selected, left_tail=left_tail)
                conf_pval_add[i, idx] = compute_conf_pvals_np(scores_aug[-1], scores_aug[:-1], left_tail=left_tail)
            C_set_y = set(labels[np.where(conf_pval_y[i] >= self.alpha)[0]])
            C_set_add = set(labels[np.where(conf_pval_add[i] >= self.alpha)[0]])
            C_sets_final.append(sorted(C_set_y.union(C_set_add)))
            k_final.append(list(set.intersection(*k_hat_i)) if k_hat_i else [])
        return (C_sets_final, k_final) if return_khat else C_sets_final
