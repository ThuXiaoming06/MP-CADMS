import numpy as np
import pandas as pd
from utils import get_args_parser
from remasker_eva2 import ReMasker
'''
X_raw = np.arange(5000).reshape(1000, 5) * 1.0
X = pd.DataFrame(X_raw, columns=['0', '1', '2', '3', '4'])
X.iat[3,0] = np.nan
#X.iat[10, 1] = np.nan
#X.iat[20, 3] = np.nan
print('X:', X)
'''

path = "IVD_labeled_Remasker.xlsx"
cols = [
    "Gender","Age","sFLC-κ","sFLC-λ","sFLC-κ/λ","U-Pro","24hUPr","α1","α2","Alb%",
    "β1","β2","γ","A/G","F-κ","F-λ","M Pro.","24hU-V","HGB","Ca","Cr(E)",
    "CRP/hsCRP","CK","NT-proBNP","UA","LD","Alb","PT","PT%","INR","Fbg",
    "APTT","APTT-R","TT","D-Dimer","β2MG"
]

# 读取：把常见缺失标记直接解析为 NaN
df = pd.read_excel(
    path, sheet_name=0, engine="openpyxl",
    na_values=["", " ", "NA", "N/A", "-", "—", "null", "None"]
).reindex(columns=cols)

# 全列转数值；无法转换的一律设为 NaN
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

# 把正负无穷也当作缺失
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 可选：显式填充（多数情况下与上面等效，这里只是强调）
df = df.fillna(np.nan)

print("shape:", df.shape)
print(df.head(3))
print(df.isna().sum().sort_values(ascending=False).head(10))
print('df:', df)

imputer = ReMasker()

imputed = imputer.fit_transform(df)
print('imputed:', imputed)

df_imputed = pd.DataFrame(imputed, columns=cols)
df_imputed.to_excel('imputed_output_IVD.xlsx', index=False, header=True)

