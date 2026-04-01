import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

def _read_csv_any(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    if os.path.exists(path + ".gz"):
        return pd.read_csv(path + ".gz", compression='gzip')
    raise FileNotFoundError(f"File not found: {path}(.gz)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mimic-root", type=str, default="dataset/mimic-iv-3.1/mimic-iv-3.1/")
    p.add_argument("--out-csv", type=str, default="dataset/mimic_iv_processed.csv")
    p.add_argument("--minority-def", type=str, default="non_white", choices=["non_white", "black_vs_rest"])
    args = p.parse_args()

    root = Path(args.mimic_root)
    hosp = root / "hosp"
    icu_dir = root / "icu"

    admissions = _read_csv_any(str(hosp / "admissions.csv"))
    patients = _read_csv_any(str(hosp / "patients.csv"))
    
    icu_path = icu_dir / "icustays.csv"
    diag_path = hosp / "diagnoses_icd.csv"
    has_icu = icu_path.exists() or Path(str(icu_path) + ".gz").exists()
    has_diag = diag_path.exists() or Path(str(diag_path) + ".gz").exists()

    req_adm = ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "race", "insurance", "admission_type", "marital_status", "hospital_expire_flag"]
    for c in req_adm:
        if c not in admissions.columns:
            raise ValueError(f"Missing admissions column: {c}")
            
    req_pat = ["subject_id", "gender", "anchor_age", "anchor_year"]
    for c in req_pat:
        if c not in patients.columns:
            raise ValueError(f"Missing patients column: {c}")

    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")
    admissions["deathtime"] = pd.to_datetime(admissions["deathtime"], errors="coerce")

    df = admissions.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left"
    )

    if has_icu:
        icu = _read_csv_any(str(icu_path))
        icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
        
        icu_agg = icu.groupby("hadm_id").agg(
            icu_los=("los", "sum"),
            first_icu_intime=("intime", "min")
        ).reset_index()
        
        df = df.merge(icu_agg, on="hadm_id", how="left")
        df["icu_los"] = df["icu_los"].fillna(0.0).astype(np.float32)
        
        # 计算入院到首次进入ICU的延迟时间(小时)
        df["time_to_icu_hours"] = (df["first_icu_intime"] - df["admittime"]).dt.total_seconds() / 3600.0
        df["time_to_icu_hours"] = df["time_to_icu_hours"].clip(lower=0.0).astype(np.float32)
    else:
        df["icu_los"] = 0.0
        df["time_to_icu_hours"] = np.nan

    if has_diag:
        diag = _read_csv_any(str(diag_path))
        diag_cnt = diag.groupby("hadm_id").size().reset_index(name="num_diagnoses")
        df = df.merge(diag_cnt, on="hadm_id", how="left")
        df["num_diagnoses"] = df["num_diagnoses"].fillna(0).astype(np.float32)
    else:
        df["num_diagnoses"] = 0.0

    df = df.dropna(subset=["admittime", "dischtime", "anchor_age", "anchor_year"]).copy()

    df["age"] = df["anchor_age"] + (df["admittime"].dt.year - df["anchor_year"])
    df["age"] = df["age"].clip(lower=0.0, upper=89.0).astype(np.float32)

    # === 标签构建 ===
    # 任务1: 死亡率
    df["label"] = df["hospital_expire_flag"].astype(np.int32)
    # 任务2: ICU延迟分类 (0: <=4h, 1: 4-12h, 2: >12h或未进)
    conditions = [
        (df["time_to_icu_hours"] <= 4.0),
        (df["time_to_icu_hours"] > 4.0) & (df["time_to_icu_hours"] <= 12.0),
        (df["time_to_icu_hours"] > 12.0) | df["time_to_icu_hours"].isna()
    ]
    df["icu_delay_class"] = np.select(conditions, [0, 1, 2], default=2).astype(np.int32)

    # === 敏感属性构建 ===
    race = df["race"].fillna("UNKNOWN").astype(str).str.upper()
    if args.minority_def == "non_white":
        minority = ~race.str.contains("WHITE", regex=False)
    else:
        minority = race.str.contains("BLACK", regex=False)
    df["minority"] = minority.astype(np.int32)

    gender = df["gender"].fillna("U").astype(str).str.upper()
    df["gender_m"] = (gender == "M").astype(np.int32)

    insurance = df["insurance"].fillna("UNKNOWN").astype(str).str.upper()
    df["ins_private"] = (insurance == "PRIVATE").astype(np.int32)
    df["ins_medicare"] = (insurance == "MEDICARE").astype(np.int32)
    df["ins_medicaid"] = (insurance == "MEDICAID").astype(np.int32)
    df["public_insurance"] = (df["ins_medicare"] | df["ins_medicaid"]).astype(np.int32)
    
    adm_type = df["admission_type"].fillna("UNKNOWN").astype(str).str.upper()
    df["adm_emergency"] = (adm_type.str.contains("EMERGENCY")).astype(np.int32)
    df["adm_elective"] = (adm_type.str.contains("ELECTIVE")).astype(np.int32)
    df["adm_urgent"] = (adm_type.str.contains("URGENT")).astype(np.int32)
    
    marital = df["marital_status"].fillna("UNKNOWN").astype(str).str.upper()
    df["marital_married"] = (marital == "MARRIED").astype(np.int32)
    df["marital_single"] = (marital == "SINGLE").astype(np.int32)

    x_cols = [
        "age", "gender_m", "ins_private", "ins_medicare", "ins_medicaid",
        "adm_emergency", "adm_elective", "adm_urgent",
        "marital_married", "marital_single",
        "icu_los", "num_diagnoses"
    ]
    keep = ["subject_id", "hadm_id", "label", "icu_delay_class", "minority", "gender_m", "public_insurance"] + x_cols
    out = df[keep].copy()
    
    out = out.rename(columns={"subject_id": "SUBJECT_ID", "hadm_id": "HADM_ID"})
    out = out.dropna(subset=x_cols + ["label", "icu_delay_class", "minority", "gender_m", "public_insurance"])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"saved to: {out_path}")
    print(f"num rows: {len(out)}")

if __name__ == "__main__":
    main()