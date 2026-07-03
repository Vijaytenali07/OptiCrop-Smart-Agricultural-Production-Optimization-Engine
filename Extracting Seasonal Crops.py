# ==========================================================
# OptiCrop: Smart Agricultural Production Optimization Engine
# Extracting Seasonal Crops
# ==========================================================

# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------
df = pd.read_csv("data/OptiCrop_Dataset.csv")   # Update the file path if required

# ----------------------------------------------------------
# Display Dataset Information
# ----------------------------------------------------------
print("Dataset Shape:", df.shape)
print("\nColumns in Dataset:")
print(df.columns.tolist())

# ----------------------------------------------------------
# Check Required Columns
# ----------------------------------------------------------
required_columns = ["Season", "Crop"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

# ==========================================================
# 1. Unique Seasons
# ==========================================================

print("\nAvailable Seasons:")
print(df["Season"].unique())

# ==========================================================
# 2. Extract Crops by Season
# ==========================================================

seasonal_crops = {}

for season in df["Season"].unique():

    crops = sorted(
        df[df["Season"] == season]["Crop"].unique()
    )

    seasonal_crops[season] = crops

# Display Results
print("\n========== Seasonal Crops ==========\n")

for season, crops in seasonal_crops.items():

    print(f"{season} Season")

    for crop in crops:
        print(f"   • {crop}")

    print("-" * 40)

# ==========================================================
# 3. Count Crops in Each Season
# ==========================================================

crop_count = df.groupby("Season")["Crop"].nunique()

print("\nNumber of Unique Crops in Each Season:")
print(crop_count)

# ==========================================================
# 4. Crop Frequency per Season
# ==========================================================

season_crop_frequency = (
    df.groupby(["Season", "Crop"])
      .size()
      .reset_index(name="Count")
)

print("\nCrop Frequency by Season:")
print(season_crop_frequency)

# ==========================================================
# 5. Visualize Crop Counts by Season
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Season",
    order=df["Season"].value_counts().index,
    palette="viridis"
)

plt.title("Number of Records by Season")
plt.xlabel("Season")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# ==========================================================
# 6. Top Crops in Each Season
# ==========================================================

for season in df["Season"].unique():

    print(f"\nTop Crops in {season} Season")

    top_crops = (
        df[df["Season"] == season]["Crop"]
        .value_counts()
        .head(10)
    )

    print(top_crops)

# ==========================================================
# 7. Save Seasonal Crop Information
# ==========================================================

season_crop_frequency.to_csv(
    "Seasonal_Crop_Summary.csv",
    index=False
)

print("\nSeasonal crop summary saved successfully!")

# ==========================================================
# 8. Summary
# ==========================================================

print("\n========== Seasonal Crop Extraction Completed ==========")
print(f"Total Seasons Found : {df['Season'].nunique()}")
print(f"Total Crops Found   : {df['Crop'].nunique()}")