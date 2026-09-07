"""
eda.py
------
Exploratory Data Analysis: generates and saves key plots to reports/figures
so they can be referenced in the project report / presentation.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style("whitegrid")


def run_eda(df):
    # 1. Class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Placement_Status", data=df, palette="Set2")
    plt.title("Placement Status Distribution (0=Not Placed, 1=Placed)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_class_balance.png"), dpi=150)
    plt.close()

    # 2. Correlation heatmap
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_correlation_heatmap.png"), dpi=150)
    plt.close()

    # 3. CGPA vs Placement
    plt.figure(figsize=(5, 4))
    sns.boxplot(x="Placement_Status", y="CGPA", data=df, palette="Set3")
    plt.title("CGPA vs Placement Status")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_cgpa_vs_placement.png"), dpi=150)
    plt.close()

    # 4. Technical skills distribution by placement
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=df, x="Technical_Skills", hue="Placement_Status", fill=True, common_norm=False)
    plt.title("Technical Skills Distribution by Placement")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_technical_skills_kde.png"), dpi=150)
    plt.close()

    # 5. Placement rate by branch
    plt.figure(figsize=(6, 4))
    branch_rate = df.groupby("Branch")["Placement_Status"].mean().sort_values(ascending=False)
    sns.barplot(x=branch_rate.index, y=branch_rate.values, palette="viridis")
    plt.ylabel("Placement Rate")
    plt.title("Placement Rate by Branch")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_placement_rate_by_branch.png"), dpi=150)
    plt.close()

    print(f"EDA figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    data_path = os.path.join(BASE_DIR, "data", "placement_data.csv")
    df = pd.read_csv(data_path)
    run_eda(df)
