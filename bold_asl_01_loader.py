"""
bold_asl_01_loader.py
Inventory builder for BOLD and ASL resting-state scans.

BOLD : data/BOLD/{sub}_rfMRI_{session}_{AP|PA}.nii.gz    (333 timepoints)
ASL  : data/ASL/swrdr63real_{sub}_{session}_{LR|RL}.nii  ( 63 timepoints)
"""

import os
import re
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def build_inventory() -> pd.DataFrame:
    """Scan data directories and return a DataFrame of all available scans."""
    records = []
    bold_dir = os.path.join(DATA_DIR, "BOLD")
    asl_dir  = os.path.join(DATA_DIR, "ASL")

    for fname in os.listdir(bold_dir):
        m = re.match(r"(\d+)_rfMRI_(REST\d+)_(AP|PA)\.nii\.gz$", fname)
        if m:
            records.append(dict(
                subject=m.group(1), session=m.group(2),
                modality="BOLD", direction=m.group(3),
                filepath=os.path.join(bold_dir, fname),
            ))

    for fname in os.listdir(asl_dir):
        m = re.match(r"swrdr63real_(\d+)_(REST\d+)_(LR|RL)\.nii$", fname)
        if m:
            records.append(dict(
                subject=m.group(1), session=m.group(2),
                modality="ASL", direction=m.group(3),
                filepath=os.path.join(asl_dir, fname),
            ))

    df = (pd.DataFrame(records)
          .sort_values(["modality", "subject", "session", "direction"])
          .reset_index(drop=True))
    return df


if __name__ == "__main__":
    df = build_inventory()
    print(df.to_string())
    for mod in ["BOLD", "ASL"]:
        sub_df = df[df.modality == mod]
        with_rest2 = sub_df.groupby("subject")["session"].nunique().eq(2).sum()
        print(f"\n{mod}: {sub_df['subject'].nunique()} subjects, "
              f"{with_rest2} with REST1+REST2, "
              f"{len(sub_df)} total scan files")
