"""
MP-CADMS SCI data preparation and imputation workflow.

This script uses ReMasker for missing-value imputation.
The ReMasker implementation is not redistributed with MP-CADMS.

Original ReMasker repository:
https://github.com/alps-lab/remasker

Reference:
Du, T., Melis, L. & Wang, T.
ReMasker: Imputing Tabular Data with Masked Autoencoding.
ICLR 2024.
"""

import numpy as np
import pandas as pd

# ReMasker must be obtained separately from its original repository.
from remasker_impute import ReMasker


path = "IVD_labeled_Remasker.xlsx"

cols = [
    "Gender", "Age", "sFLC-κ", "sFLC-λ", "sFLC-κ/λ", "U-Pro", "24hUPr",
    "α1", "α2", "Alb%", "β1", "β2", "γ", "A/G", "F-κ", "F-λ",
    "M Pro.", "24hU-V", "HGB", "Ca", "Cr(E)", "CRP/hsCRP", "CK",
    "NT-proBNP", "UA", "LD", "Alb", "PT", "PT%", "INR", "Fbg",
    "APTT", "APTT-R", "TT", "D-Dimer", "β2MG"
]

# Load SCI data and convert common missing-value markers to NaN.
df = pd.read_excel(
    path,
    sheet_name=0,
    engine="openpyxl",
    na_values=["", " ", "NA", "N/A", "-", "—", "null", "None"]
).reindex(columns=cols)

# Convert all SCI variables to numeric values.
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

# Treat positive and negative infinity as missing values.
df.replace([np.inf, -np.inf], np.nan, inplace=True)

imputer = ReMasker()
imputed = imputer.fit_transform(df)

df_imputed = pd.DataFrame(imputed, columns=cols)
df_imputed.to_excel(
    "imputed_output_IVD.xlsx",
    index=False,
    header=True
)
