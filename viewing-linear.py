# %%
import pandas as pd
import numpy as np

# Seting all float values to be shown with 4 algarisms after comma
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

# %%

df_alq = pd.read_sas("data/ALQ_L_2021_2023.xpt")
df_alq.to_csv("data/ALQ_L_2021_2023.csv", index=False)
df_alq


# %%

df_bmx = pd.read_sas("data/BMX_L_2021_2023.xpt")
df_bmx.to_csv("data/BMX_L_2021_2023.csv", index=False)
df_bmx

# %%

df_bpq = pd.read_sas("data/BPQ_L_2021_2023.xpt")
df_bpq.to_csv("data/BPQ_L_2021_2023.csv", index=False)
df_bpq

# %%

df_bpxo = pd.read_sas("data/BPXO_L_2021_2023.xpt")
df_bpxo.to_csv("data/BPXO_L_2021_2023.csv", index=False)
df_bpxo

# %%

df_demo = pd.read_sas("data/DEMO_L_2021_2023.xpt")
df_demo.to_csv("data/DEMO_L_2021_2023.csv", index=False)
df_demo

# %%

df_diq = pd.read_sas("data/DIQ_L_2021_2023.xpt")
df_diq.to_csv("data/DIQ_L_2021_2023.csv", index=False)
df_diq

# %%

df_dpq = pd.read_sas("data/DPQ_L_2021_2023.xpt")
df_dpq.to_csv("data/DPQ_L_2021_2023.csv", index=False)
df_dpq

# %%

df_huq = pd.read_sas("data/HUQ_L_2021_2023.xpt")
df_huq.to_csv("data/HUQ_L_2021_2023.csv", index=False)
df_huq


# %%

df_paq = pd.read_sas("data/PAQ_L_2021_2023.xpt")
df_paq.to_csv("data/PAQ_L_2021_2023.csv", index=False)
df_paq

# %%

df_slq = pd.read_sas("data/SLQ_L_2021_2023.xpt")
df_slq.to_csv("data/SLQ_L_2021_2023.csv", index=False)
df_slq

# %%

df_whq = pd.read_sas("data/WHQ_L_2021_2023.xpt")
df_whq.to_csv("data/WHQ_L_2021_2023.csv", index=False)
df_whq

# %% 
# From now on the cells will be focused on manipulating the dataframes based on the documents files

# Creating another df just to manipulate it instead of the original
df_filtered_alq = df_alq[["SEQN", "ALQ151"]].copy()
df_filtered_alq = df_filtered_alq.set_index("SEQN")
new_columns_alq = {"ALQ151" : "Ever have 4+ drinks every day"}
df_filtered_alq.rename(columns=new_columns_alq, inplace=True)
df_filtered_alq["Ever have 4+ drinks every day"] = df_filtered_alq["Ever have 4+ drinks every day"].map({2: 0, 1: 1})
df_filtered_alq

# %%

df_filtered_bmx = df_bmx[["SEQN", "BMXBMI", "BMXWAIST", "BMXHIP"]].copy()
df_filtered_bmx = df_filtered_bmx.set_index("SEQN")
new_columns_bmx = {"BMXBMI" : "Body Mass Index"}
df_filtered_bmx.rename(columns=new_columns_bmx, inplace=True)
df_filtered_bmx["Waist-to-Hip Ratio"] = df_filtered_bmx["BMXWAIST"] / df_filtered_bmx["BMXHIP"]
# I'm droping the WHR's primitives columns because I'll first train linear models, as Logistic Regression and SVMs
df_filtered_bmx = df_filtered_bmx.drop(columns=["BMXWAIST", "BMXHIP"])
df_filtered_bmx

# %%

df_filtered_bpq = df_bpq[["SEQN", "BPQ020", "BPQ080", "BPQ101D"]].copy()
df_filtered_bpq = df_filtered_bpq.set_index("SEQN")
# For linear models, I'll drop features that might be collinear
# For linear models, creating new binary features that are mutually exclusive
df_filtered_bpq["High Cholesterol WITHOUT Medicine"] = np.where((df_filtered_bpq["BPQ080"] == 1) & (df_filtered_bpq["BPQ101D"] == 2), 1, 0)
df_filtered_bpq["High Colesterol WITH Medicine"] = np.where((df_filtered_bpq["BPQ080"] == 1) & (df_filtered_bpq["BPQ101D"] == 1), 1, 0)
df_filtered_bpq["BPQ020"] = df_filtered_bpq["BPQ020"].map({2 : 0, 1 : 1})
df_filtered_bpq = df_filtered_bpq.drop(columns=["BPQ080", "BPQ101D"])
new_columns_bpq = {"SEQN" : "SEQN", "BPQ020" : "Have Hypertension"}
df_filtered_bpq.rename(columns=new_columns_bpq, inplace=True)
df_filtered_bpq

# %%

# For linear models, I'll use only the systolic pressure to avoid collinearity. For non-linear models I might use the pulse and dyastolic pressure also
df_filtered_bpxo = df_bpxo[["SEQN", "BPXOSY1", "BPXOSY2", "BPXOSY3"]].copy()
df_filtered_bpxo = df_filtered_bpxo.set_index("SEQN")
df_filtered_bpxo["Mean Systolic Pressure"] =  df_filtered_bpxo[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
df_filtered_bpxo = df_filtered_bpxo.drop(columns=["BPXOSY1", "BPXOSY2", "BPXOSY3"])
df_filtered_bpxo

# %%

df_filtered_demo = df_demo[["SEQN", "RIAGENDR", "RIDAGEYR", "INDFMPIR"]].copy()
df_filtered_demo = df_filtered_demo.set_index("SEQN")
new_columns_demo = {"RIAGENDR" : "Gender(Male)", "RIDAGEYR" : "Age", "INDFMPIR" : "Ratio of Family Income to Poverty"}
df_filtered_demo.rename(columns=new_columns_demo, inplace=True)
df_filtered_demo["Gender(Male)"] = df_filtered_demo["Gender(Male)"].map({2 : 0, 1 : 1})
df_filtered_demo

# %%

df_filtered_diq = df_diq[["SEQN", "DIQ010"]].copy()
df_filtered_diq = df_filtered_diq.set_index("SEQN")
df_filtered_diq["DIQ010"] = df_filtered_diq["DIQ010"].map({2 : 0, 1 : 1})
# For now, the target ("Has Diabetes?") will be determined only by DIQ010, but soon, when I add laboratory data to the dfs, it'll change
df_filtered_diq.rename(columns={"DIQ010" : "Has Diabetes"}, inplace=True)
df_filtered_diq

# %%

# I'll resume all features from this df in one only feature that will be the sum of each primitive feature
df_filtered_dpq = df_dpq[["SEQN", "DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080", "DPQ090", "DPQ100"]].copy()
df_filtered_dpq = df_filtered_dpq.set_index("SEQN")
# For this specific case, replace has a good utility, because all "Refused"(7) or "Don't Know"(9) will be replaced as NaNs
df_filtered_dpq = df_filtered_dpq.replace([7, 9], np.nan)
df_filtered_dpq["Mental Health"] = df_filtered_dpq.loc[:, "DPQ010":"DPQ100"].sum(axis=1, skipna=False)
dropped_columns_dpq = ["DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080", "DPQ090", "DPQ100"]
df_filtered_dpq.drop(columns=dropped_columns_dpq, inplace=True)
df_filtered_dpq


# %%

df_filtered_huq = df_huq[["SEQN", "HUQ010"]].copy()
df_filtered_huq = df_filtered_huq.set_index("SEQN")
df_filtered_huq["General Health Condition (Self-evaluated)"] = df_filtered_huq["HUQ010"].map({1 : 1, 2 : 1, 3 : 1, 4 : 0, 5 : 0})
df_filtered_huq.drop(columns="HUQ010", inplace=True)
df_filtered_huq

# %%

df_filtered_paq = df_paq[["SEQN", "PAD790Q", "PAD790U", "PAD800", "PAD810Q", "PAD810U", "PAD820", "PAD680"]].copy()
df_filtered_paq = df_filtered_paq.set_index("SEQN")
# This seems to be an exception for what I have done until here (don't droping NaNs)
# This specifics features (PAD790Q and PAD680) only has 18 NaNs, so I decided to drop this NaNs and set all other NaNs as 0
# More about how and why I decided to do that in README.md
df_filtered_paq.dropna(subset=["PAD790Q", "PAD680"], inplace=True)
df_filtered_paq = df_filtered_paq.replace(np.nan, 0)
df_filtered_paq = df_filtered_paq.replace([7777, 9999], np.nan)
# Because the unities in PAD790Q and PAD810Q are encoded by it's initial (example: month = M)
# I decided to transform the time measure in minutes/week, 
# and remaped the encoded initials for the factor between week and the time measure the patient chose to use
remaping_unity = {b'D' : 7, b'W' : 1, b'M' : 0.2304, b'Y': 0.0192, b'' : 0}
time_factor_moderate_paq = df_filtered_paq["PAD790U"].map(remaping_unity)
time_factor_vigorous_paq = df_filtered_paq["PAD810U"].map(remaping_unity)
# Thinking about create one only feature that could represent all those combined, 
# I decided to multiply minutes/session by session/week, obtaining minutes/week
moderate_ltpa_minutes = df_filtered_paq["PAD790Q"] * df_filtered_paq["PAD800"] * time_factor_moderate_paq
vigorous_ltpa_minutes = df_filtered_paq["PAD810Q"] * df_filtered_paq["PAD820"] * time_factor_vigorous_paq
# Commentarys about the line of code right below in README.md
df_filtered_paq["Total LTPA (minutes/week)"] = moderate_ltpa_minutes + 2 * vigorous_ltpa_minutes
df_filtered_paq["Sedentary activity (minutes/week)"] = df_filtered_paq["PAD680"] * 7
df_filtered_paq.drop(columns=["PAD790Q", "PAD790U", "PAD800", "PAD810Q", "PAD810U", "PAD820", "PAD680"], inplace=True)
df_filtered_paq
# %%

df_filtered_slq = df_slq[["SEQN", "SLD012", "SLD013"]].copy()
df_filtered_slq = df_filtered_slq.set_index("SEQN")
df_filtered_slq["Mean Sleep Time"] = df_filtered_slq[["SLD012", "SLD013"]].mean(axis=1)
df_filtered_slq.drop(columns=["SLD012", "SLD013"], inplace=True)
df_filtered_slq

# %% 
df_filtered_whq = df_whq[["SEQN", "WHD050", "WHQ070"]].copy()
df_filtered_whq = df_filtered_whq.set_index("SEQN")
df_weight = df_bmx[["SEQN", "BMXWT"]]
df_weight = df_weight.set_index("SEQN")
# I divided the WHD050 column by 2.20462 because this feature is measured in kg but WHD050 is measured in pounds
df_filtered_whq["Weight Variation Within 1 Year"] = df_weight["BMXWT"] - df_filtered_whq["WHD050"] / 2.20462
df_filtered_whq["Tried to Lose Weight Within 1 Year"] = df_filtered_whq["WHQ070"].map({2 : 0, 1 : 1})
df_filtered_whq.drop(columns=["WHD050", "WHQ070"], inplace=True)
df_filtered_whq

