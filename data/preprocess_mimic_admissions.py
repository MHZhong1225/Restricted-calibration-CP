import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _read_csv_any(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    if os.path.exists(path + ".gz"):
        return pd.read_csv(path + ".gz")
    raise FileNotFoundError(f"File not found: {path}(.gz)")


def _safe_age_years(admit: pd.Series, dob: pd.Series) -> pd.Series:
    """
    Avoid pandas datetime subtraction overflow by computing age from Y/M/D fields.
    """
    admit = pd.to_datetime(admit, errors="coerce")
    dob = pd.to_datetime(dob, errors="coerce")

    age = admit.dt.year - dob.dt.year

    # birthday not reached yet this year -> subtract 1
    before_birthday = (
        (admit.dt.month < dob.dt.month) |
        ((admit.dt.month == dob.dt.month) & (admit.dt.day < dob.dt.day))
    )
    age = age - before_birthday.astype("Int64")

    return age.astype("float32")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mimic-root", type=str, default="data/mimic-iii-clinical-database-1.4/")
    p.add_argument("--out-csv", type=str, default="data/mimic_admissions_processed.csv")
    p.add_argument("--minority-def", type=str, default="non_white", choices=["non_white", "black_vs_rest"])
    args = p.parse_args()

    root = Path(args.mimic_root)
    admissions = _read_csv_any(str(root / "ADMISSIONS.csv"))
    patients = _read_csv_any(str(root / "PATIENTS.csv"))
    
    # Optional files for more features
    icu_path = root / "ICUSTAYS.csv"
    diag_path = root / "DIAGNOSES_ICD.csv"
    has_icu = icu_path.exists() or Path(str(icu_path) + ".gz").exists()
    has_diag = diag_path.exists() or Path(str(diag_path) + ".gz").exists()

    for c in ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "ETHNICITY", "INSURANCE", "ADMISSION_TYPE", "MARITAL_STATUS"]:
        if c not in admissions.columns:
            raise ValueError(f"Missing ADMISSIONS column: {c}")
    for c in ["SUBJECT_ID", "GENDER", "DOB"]:
        if c not in patients.columns:
            raise ValueError(f"Missing PATIENTS column: {c}")

    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce")
    admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"], errors="coerce")
    admissions["DEATHTIME"] = pd.to_datetime(admissions["DEATHTIME"], errors="coerce")
    patients["DOB"] = pd.to_datetime(patients["DOB"], errors="coerce")

    df = admissions.merge(
        patients[["SUBJECT_ID", "GENDER", "DOB"]],
        on="SUBJECT_ID",
        how="left"
    )

    if has_icu:
        icu = _read_csv_any(str(icu_path))
        icu_agg = icu.groupby("HADM_ID")["LOS"].sum().reset_index().rename(columns={"LOS": "icu_los"})
        df = df.merge(icu_agg, on="HADM_ID", how="left")
        df["icu_los"] = df["icu_los"].fillna(0.0).astype(np.float32)
    else:
        df["icu_los"] = 0.0

    if has_diag:
        diag = _read_csv_any(str(diag_path))
        diag_cnt = diag.groupby("HADM_ID").size().reset_index(name="num_diagnoses")
        df = df.merge(diag_cnt, on="HADM_ID", how="left")
        df["num_diagnoses"] = df["num_diagnoses"].fillna(0).astype(np.float32)
    else:
        df["num_diagnoses"] = 0.0

    df = df.dropna(subset=["ADMITTIME", "DISCHTIME", "DOB"]).copy()

    # safe age computation
    df["age"] = _safe_age_years(df["ADMITTIME"], df["DOB"])

    # remove impossible ages
    df = df[df["age"].notna()].copy()
    df = df[df["age"] >= 0].copy()

    # MIMIC-III de-identification: age > 89 is masked, commonly clipped
    df["age"] = df["age"].clip(lower=0.0, upper=89.0).astype(np.float32)

    y = (~df["DEATHTIME"].isna()) & (df["DEATHTIME"] <= df["DISCHTIME"])
    df["label"] = y.astype(np.int32)

    eth = df["ETHNICITY"].fillna("UNKNOWN").astype(str).str.upper()
    df["ethnicity_raw"] = eth

    if args.minority_def == "non_white":
        minority = ~eth.str.contains("WHITE", regex=False)
    else:
        minority = eth.str.contains("BLACK", regex=False)
    df["minority"] = minority.astype(np.int32)

    gender = df["GENDER"].fillna("U").astype(str).str.upper()
    df["gender_m"] = (gender == "M").astype(np.int32)

    insurance = df["INSURANCE"].fillna("UNKNOWN").astype(str).str.upper()
    df["ins_private"] = (insurance == "PRIVATE").astype(np.int32)
    df["ins_medicare"] = (insurance == "MEDICARE").astype(np.int32)
    df["ins_medicaid"] = (insurance == "MEDICAID").astype(np.int32)
    
    adm_type = df["ADMISSION_TYPE"].fillna("UNKNOWN").astype(str).str.upper()
    df["adm_emergency"] = (adm_type == "EMERGENCY").astype(np.int32)
    df["adm_elective"] = (adm_type == "ELECTIVE").astype(np.int32)
    df["adm_urgent"] = (adm_type == "URGENT").astype(np.int32)
    
    marital = df["MARITAL_STATUS"].fillna("UNKNOWN").astype(str).str.upper()
    df["marital_married"] = (marital == "MARRIED").astype(np.int32)
    df["marital_single"] = (marital == "SINGLE").astype(np.int32)

    x_cols = [
        "age", "gender_m", "ins_private", "ins_medicare", "ins_medicaid",
        "adm_emergency", "adm_elective", "adm_urgent",
        "marital_married", "marital_single",
        "icu_los", "num_diagnoses"
    ]
    keep = ["SUBJECT_ID", "HADM_ID", "label", "minority"] + x_cols
    out = df[keep].copy()
    out = out.dropna(subset=x_cols + ["label", "minority"])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"saved to: {out_path}")
    print(f"num rows: {len(out)}")


if __name__ == "__main__":
    main()