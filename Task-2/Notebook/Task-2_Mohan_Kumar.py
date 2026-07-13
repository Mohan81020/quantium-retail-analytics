#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Load cleaned dataset

df_final = pd.read_csv("E:\Quanties_Data_Analyst_Project\Task-2\QVI_data.csv")


# In[3]:


df_final.head()


# In[4]:


df_final.rename(columns = str.lower,inplace = True)


# In[5]:


df_final.head(2)


# In[6]:


# Convert date column into datetime format

df_final["date"] = pd.to_datetime(
    df_final["date"],
    format="%d-%m-%Y"
)
df_final['month_id'] = df_final['date'].dt.to_period('M')


# In[7]:


df_final.head()


# In[8]:


"""Create Monthly Metrics
1. Total Sales
2. Number of Customers
3. Average Transactions per Customer

We aggregate store-month level"""


# In[9]:


df_final.columns


# In[10]:


# Aggregate monthly metrics for each store

monthly_matrix = df_final.groupby(['store_nbr','month_id']).agg(
Total_sales = ('tot_sales','sum'),
no_of_cust = ('lylty_card_nbr','nunique'),
transaction = ('txn_id','nunique')).reset_index()

monthly_matrix['Avg_txn_per_cust'] =  monthly_matrix['transaction']/monthly_matrix['no_of_cust']

monthly_matrix


# In[11]:


monthly_matrix.describe()


# In[12]:


# Define Pre-Trial Period
# trail start from Feb'2019 so control period will be before feb'19 


# In[13]:


pre_trial = monthly_matrix[monthly_matrix['month_id'] < '2019-02']
pre_trial.head()


# In[14]:


from scipy.stats import pearsonr
import numpy as np
import pandas as pd

def calculate_correlation(metric, trial_store):

    # Data for the trial store
    trial = pre_trial[
        pre_trial["store_nbr"] == trial_store
    ][["month_id", metric]]

    correlations = {}

    # Compare with every other store
    for store in pre_trial["store_nbr"].unique():

        if store == trial_store:
            continue

        control = pre_trial[
            pre_trial["store_nbr"] == store
        ][["month_id", metric]]

        # Merge on month
        merged = pd.merge(
            trial,
            control,
            on="month_id",
            suffixes=("_trial", "_control")
        )

        # Skip stores with missing months
        if len(merged) != len(trial):
            continue

        trial_values = merged[f"{metric}_trial"]
        control_values = merged[f"{metric}_control"]

        # Pearson correlation cannot be computed if values are constant
        if trial_values.nunique() <= 1 or control_values.nunique() <= 1:
            correlations[store] = np.nan
        else:
            corr, _ = pearsonr(trial_values, control_values)
            correlations[store] = corr

    return pd.Series(correlations, name=f"{metric}_corr")


# In[15]:


Avg_txn_per_cust_corr = calculate_correlation('Avg_txn_per_cust',77)
Avg_txn_per_cust_corr.sort_values(ascending=False).head(10)


# In[16]:


# total sales correlations

sales_corr = calculate_correlation('Total_sales',77)
sales_corr.sort_values(ascending=False).head(10)


# In[17]:


# no_of_cust correlation

no_of_cust_corr = calculate_correlation('no_of_cust',77)
no_of_cust_corr.sort_values(ascending=False).head(10)


# In[18]:


monthly_matrix.head()


# In[19]:


pre_trial.columns


# In[20]:


def calculate_magnitude(metric, trial_store):

    # Data for the trial store
    trial = pre_trial[
        pre_trial["store_nbr"] == trial_store
    ][["month_id", metric]]

    magnitude = {}

    # Compare with every other store
    for store in pre_trial["store_nbr"].unique():

        if store == trial_store:
            continue

        control = pre_trial[
            pre_trial["store_nbr"] == store
        ][["month_id", metric]]

        merged = pd.merge(
            trial,
            control,
            on="month_id",
            suffixes=("_trial", "_control")
        )

        # Skip stores with missing months
        if len(merged) != len(trial):
            continue

        diff = abs(
            merged[f"{metric}_trial"] -
            merged[f"{metric}_control"]
        )

        # Handle constant differences
        if diff.max() == diff.min():
            magnitude_score = 1
        else:
            score = 1 - (
                (diff - diff.min()) /
                (diff.max() - diff.min())
            )
            magnitude_score = score.mean()

        magnitude[store] = magnitude_score

    return pd.Series(magnitude, name=f"{metric}_magn")


# In[21]:


Total_sales_mag = calculate_magnitude('Total_sales',77)
Total_sales_mag.sort_values(ascending = False).head(10)


# In[22]:


no_of_cust_mag = calculate_magnitude('no_of_cust',77)
no_of_cust_mag.sort_values(ascending = False).head(10)


# In[23]:


Avg_txn_per_cust_mag = calculate_magnitude('Avg_txn_per_cust',77)
Avg_txn_per_cust_mag.sort_values(ascending = False).head(10)


# In[24]:


monthly_matrix.columns


# In[25]:


def get_similarity(trial_store):

    sales_corr = calculate_correlation("Total_sales", trial_store)
    customer_corr = calculate_correlation("no_of_cust", trial_store)
    txn_corr = calculate_correlation("Avg_txn_per_cust", trial_store)

    sales_magn = calculate_magnitude("Total_sales", trial_store)
    customer_magn = calculate_magnitude("no_of_cust", trial_store)
    txn_magn = calculate_magnitude("Avg_txn_per_cust", trial_store)

    similarity = pd.concat(
        [
            sales_corr,
            customer_corr,
            txn_corr,
            sales_magn,
            customer_magn,
            txn_magn,
        ],
        axis=1
    )

    similarity.columns = [
        "sales_corr",
        "customer_corr",
        "txn_corr",
        "sales_magn",
        "customer_magn",
        "txn_magn"
    ]

    # Remove stores with undefined transaction correlation
    similarity = similarity.dropna(subset=["txn_corr"])

    # Convert index into a column
    similarity = similarity.reset_index()
    similarity.rename(columns={"index": "store_nbr"}, inplace=True)

    # Calculate average score for each metric
    similarity["sales_score"] = (
        similarity["sales_corr"] +
        similarity["sales_magn"]
    ) / 2

    similarity["customer_score"] = (
        similarity["customer_corr"] +
        similarity["customer_magn"]
    ) / 2

    similarity["txn_score"] = (
        similarity["txn_corr"] +
        similarity["txn_magn"]
    ) / 2

    # Overall similarity score
    similarity["Final_Score"] = similarity[
        ["sales_score", "customer_score", "txn_score"]
    ].mean(axis=1)

    return similarity.sort_values(
        "Final_Score",
        ascending=False
    )


# In[26]:


similarity_77 = get_similarity(77)
similarity_86 = get_similarity(86)
similarity_88 = get_similarity(88)


# In[27]:


similarity_88.head(10)


# In[28]:


import seaborn as sns
def visual_similarity(trial_store):

    top5 = (
        get_similarity(trial_store)
        .sort_values("Final_Score", ascending=False)
        .head()
    )

    plt.figure(figsize=(8,5))

    sns.barplot(
        data=top5.reset_index(),
        x="store_nbr",
        y="Final_Score"
    )

    plt.title(f"Top 5 Similar Stores for Trial Store {trial_store}")
    plt.xlabel("Store Number")
    plt.ylabel("Final Similarity Score")
    

    plt.tight_layout()
    plt.show()

    return top5


# In[29]:


top5_77 = visual_similarity(77)
top5_77


# In[30]:


top5_86 = visual_similarity(86)
top5_86


# In[ ]:





# In[31]:


top5_88 = visual_similarity(88)
top5_88


# In[32]:


def control_store_find(trial_store):
    control_score = get_similarity(trial_store).iloc[0]["store_nbr"]
    #print(f"Trial Store {trial_store} -> Control Store:, {control_score}")
    return control_score


# In[33]:


control_store_find(77)


# In[34]:


def compare_trial_control(trial_store):
    control_store = control_store_find(trial_store)
    trial = monthly_matrix[
        monthly_matrix["store_nbr"] == trial_store
    ]

    control = monthly_matrix[
        monthly_matrix["store_nbr"] == control_store
    ]

    comparison = trial.merge(
        control,
        on="month_id",
        suffixes=("_trial", "_control")
    )

    return comparison


# In[35]:


comparison_77 = compare_trial_control(77)
comparison_77.head()


# In[36]:


comparison_86 = compare_trial_control(86)
comparison_86.head()


# In[37]:


def plot_trial_control_metric(trial_store, metric, title, ylabel):

    comparison = compare_trial_control(trial_store)
    control_store = control_store_find(trial_store)

    plt.figure(figsize=(12,6))

    plt.plot(
        comparison["month_id"].astype(str),
        comparison[f"{metric}_trial"],
        marker="o",
        linewidth=2,
        label=f"Trial Store {trial_store}"
    )

    plt.plot(
        comparison["month_id"].astype(str),
        comparison[f"{metric}_control"],
        marker="o",
        linewidth=2,
        label=f"Control Store {control_store}"
    )

    plt.axvline(
        x=6,
        color="black",
        linestyle="--",
        label="Trial Begins"
    )

    plt.title(
        f"{title}\nTrial Store {trial_store} vs Control Store {control_store}"
    )

    plt.xlabel("Month")
    plt.ylabel(ylabel)

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(
    f"Store_{trial_store}_{metric}.png",
    dpi=300,
    bbox_inches="tight")

    plt.show()


# In[ ]:





# In[38]:


for store in [77, 86, 88]:

    plot_trial_control_metric(
        store,
        "Total_sales",
        "Monthly Total Sales",
        "Sales"
    )

    plot_trial_control_metric(
        store,
        "no_of_cust",
        "Monthly Number of Customers",
        "Customers"
    )

    plot_trial_control_metric(
        store,
        "Avg_txn_per_cust",
        "Monthly Average Transactions per Customer",
        "Transactions per Customer"
    )


# In[ ]:





# In[ ]:





# In[39]:


def calculate_scaling_sales(trial_store, metric):

    comparison = compare_trial_control(trial_store)

    pre_trial = comparison[
        comparison["month_id"] < "2019-02"
    ]

    scaling_factor = (
        pre_trial[f"{metric}_trial"].sum()
        /
        pre_trial[f"{metric}_control"].sum()
    )

    comparison[f"Scaled_{metric}_control"] = (
        comparison[f"{metric}_control"] *
        scaling_factor
    )

    return comparison, scaling_factor


# In[ ]:





# In[40]:


comparison_77, scaling_factor = calculate_scaling_sales(
    77,
    "Total_sales"
)


# In[41]:


comparison_77.head()


# In[42]:


comparison_86, scaling_factor = calculate_scaling_sales(
    86,
    "Total_sales"
)
comparison_86.head()


# In[43]:


comparison, scaling_factor = calculate_scaling_sales(
    77,
    "Total_sales"
)

print(comparison.columns.tolist())


# In[44]:


comparison, scaling_factor = calculate_scaling_sales(
    77,
    "Total_sales"
)

comparison["month_label"] = comparison["month_id"].astype(str)

plt.figure(figsize=(12,6))

plt.plot(
    comparison["month_label"],
    comparison["Total_sales_trial"],
    marker="o",
    label="Trial"
)

plt.plot(
    comparison["month_label"],
    comparison["Scaled_Total_sales_control"],
    marker="o",
    label="Scaled Control"
)
plt.title("Trial vs. Control Sales for Store -77")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"Store_77_T_C_sales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[ ]:





# In[45]:


# function for statistical test

def significance_test(trial_store):

    comparison, scaling_factor = calculate_scaling_sales(
        trial_store,
        "Total_sales"
    )
    # Pre-Trial Period

    pre_trial = comparison[
        comparison["month_id"] < "2019-02"
    ].copy()

    pre_trial["pct_diff"] = (
        pre_trial["Total_sales_trial"]
        -
        pre_trial["Scaled_Total_sales_control"]
    ) / pre_trial["Scaled_Total_sales_control"]

    mean_diff = pre_trial["pct_diff"].mean()

    std_diff = pre_trial["pct_diff"].std()

    upper_limit = mean_diff + 2 * std_diff

    lower_limit = mean_diff - 2 * std_diff

    # Trial Period

    trial_period = comparison[
        comparison["month_id"] >= "2019-02"
    ].copy()

    trial_period["pct_diff"] = (
        trial_period["Total_sales_trial"]
        -
        trial_period["Scaled_Total_sales_control"]
    ) / trial_period["Scaled_Total_sales_control"]

    trial_period["Significant"] = (
        (trial_period["pct_diff"] > upper_limit) |
        (trial_period["pct_diff"] < lower_limit)
    )

    return trial_period[
        [
            "month_id",
            "Total_sales_trial",
            "Scaled_Total_sales_control",
            "pct_diff",
            "Significant"
        ]
    ]


# In[46]:


significance_test(77)


# In[47]:


significance_test(86)


# In[48]:


def visualize_significance(trial_store):

    comparison, scaling_factor = calculate_scaling_sales(
        trial_store,
        "Total_sales"
    )

    control_store = control_store_find(trial_store)

    # -----------------------
    # Pre-Trial

    pre_trial = comparison[
        comparison["month_id"] < "2019-02"
    ].copy()

    pre_trial["pct_diff"] = (
        pre_trial["Total_sales_trial"]
        -
        pre_trial["Scaled_Total_sales_control"]
    ) / pre_trial["Scaled_Total_sales_control"]

    mean_diff = pre_trial["pct_diff"].mean()
    std_diff = pre_trial["pct_diff"].std()

    upper_limit = mean_diff + 2 * std_diff
    lower_limit = mean_diff - 2 * std_diff

    # -----------------------
    # Confidence Interval

    comparison["Upper_Bound"] = (
        comparison["Scaled_Total_sales_control"] *
        (1 + upper_limit)
    )

    comparison["Lower_Bound"] = (
        comparison["Scaled_Total_sales_control"] *
        (1 + lower_limit)
    )

    comparison["month_label"] = comparison["month_id"].astype(str)

    # Find trial start dynamically
    
    trial_start = comparison[
        comparison["month_label"] == "2019-02"
    ].index[0]

    # Significant months
    trial_period = comparison[
        comparison["month_id"] >= "2019-02"
    ].copy()

    significant = trial_period[
        (trial_period["Total_sales_trial"] >
         trial_period["Upper_Bound"]) |
        (trial_period["Total_sales_trial"] <
         trial_period["Lower_Bound"])
    ]

    # -----------------------
    # Plot

    plt.figure(figsize=(14,6))

    plt.plot(
        comparison["month_label"],
        comparison["Total_sales_trial"],
        marker="o",
        linewidth=2,
        label=f"Trial Store {trial_store}"
    )

    plt.plot(
        comparison["month_label"],
        comparison["Scaled_Total_sales_control"],
        marker="o",
        linewidth=2,
        label=f"Scaled Control Store {control_store}"
    )

    plt.fill_between(
        comparison["month_label"],
        comparison["Lower_Bound"],
        comparison["Upper_Bound"],
        alpha=0.25,
        label="95% Confidence Interval"
    )

    plt.scatter(
        significant["month_label"],
        significant["Total_sales_trial"],
        color="red",
        s=120,
        zorder=5,
        label="Significant Month"
    )

    plt.axvline(
        x=trial_start,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Trial Begins"
    )

    plt.title(
        f"Sales Performance Comparison\n"
        f"Trial Store {trial_store} vs Control Store {control_store}"
    )

    plt.xlabel("Month")
    plt.ylabel("Total Sales")

    plt.xticks(rotation=45)

    plt.legend()

    plt.tight_layout()
    plt.savefig(
    f"Store_{trial_store}_stats.png",
    dpi=300,
    bbox_inches="tight")

    plt.show()

    return comparison


# In[49]:


for store in [77,86,88]:
    visualize_significance(store)


# In[50]:


def final_summary(trial_store):

    # ---------------------------------------------------------
    # Get Control Store
    
    control_store = control_store_find(trial_store)

    # ---------------------------------------------------------
    # Get Comparison Data with Scaled Sales
    
    comparison, scaling_factor = calculate_scaling_sales(
        trial_store,
        "Total_sales"
    )

    # ---------------------------------------------------------
    # Pre-Trial Period
    # ---------------------------------------------------------
    pre_trial = comparison[
        comparison["month_id"] < "2019-02"
    ].copy()

    pre_trial["pct_diff"] = (
        pre_trial["Total_sales_trial"]
        -
        pre_trial["Scaled_Total_sales_control"]
    ) / pre_trial["Scaled_Total_sales_control"]

    mean_diff = pre_trial["pct_diff"].mean()
    std_diff = pre_trial["pct_diff"].std()

    upper_limit = mean_diff + (2 * std_diff)
    lower_limit = mean_diff - (2 * std_diff)

    # ---------------------------------------------------------
    # Confidence Interval
    # ---------------------------------------------------------
    comparison["Upper_Bound"] = (
        comparison["Scaled_Total_sales_control"]
        * (1 + upper_limit)
    )

    comparison["Lower_Bound"] = (
        comparison["Scaled_Total_sales_control"]
        * (1 + lower_limit)
    )

    # ---------------------------------------------------------
    # Trial Period
    # ---------------------------------------------------------
    trial_period = comparison[
        comparison["month_id"] >= "2019-02"
    ].copy()

    # ---------------------------------------------------------
    # Sales Uplift
    # ---------------------------------------------------------
    trial_period["Sales_Uplift_%"] = (
        (
            trial_period["Total_sales_trial"]
            -
            trial_period["Scaled_Total_sales_control"]
        )
        /
        trial_period["Scaled_Total_sales_control"]
    ) * 100

    # ---------------------------------------------------------
    # Significant Months
    # ---------------------------------------------------------
    significant_months = trial_period[
        (
            trial_period["Total_sales_trial"]
            > trial_period["Upper_Bound"]
        )
        |
        (
            trial_period["Total_sales_trial"]
            < trial_period["Lower_Bound"]
        )
    ]

    # ---------------------------------------------------------
    # Customer Uplift
    # ---------------------------------------------------------
    customer_uplift = (
        (
            trial_period["no_of_cust_trial"].mean()
            -
            trial_period["no_of_cust_control"].mean()
        )
        /
        trial_period["no_of_cust_control"].mean()
    ) * 100

    # ---------------------------------------------------------
    # Avg Transaction Uplift
    # ---------------------------------------------------------
    txn_uplift = (
        (
            trial_period["Avg_txn_per_cust_trial"].mean()
            -
            trial_period["Avg_txn_per_cust_control"].mean()
        )
        /
        trial_period["Avg_txn_per_cust_control"].mean()
    ) * 100

    # ---------------------------------------------------------
    # Average Sales Uplift
    # ---------------------------------------------------------
    sales_uplift = trial_period["Sales_Uplift_%"].mean()

    # ---------------------------------------------------------
    # Main Driver
    # ---------------------------------------------------------
    if abs(customer_uplift) > abs(txn_uplift):
        driver = "Increase in Customer Numbers"
    else:
        driver = "Increase in Average Transactions per Customer"

    # ---------------------------------------------------------
    # Recommendation
    # ---------------------------------------------------------
    if len(significant_months) >= 2:
        recommendation = (
            "The trial layout was successful. "
            "Roll out the layout to similar stores."
        )
    else:
        recommendation = (
            "The trial layout did not show a statistically "
            "Rollout is not recommended."
        )

    # ---------------------------------------------------------
    # Print Business Summary
    # ---------------------------------------------------------
    print("=" * 70)
    print(f"TRIAL STORE   : {trial_store}")
    print(f"CONTROL STORE : {control_store}")
    print("=" * 70)

    print(f"Scaling Factor                : {scaling_factor:.3f}")
    print(f"Average Sales Uplift          : {sales_uplift:.2f}%")
    print(f"Average Customer Uplift       : {customer_uplift:.2f}%")
    print(f"Average Transaction Uplift    : {txn_uplift:.2f}%")
    print(f"Significant Trial Months      : {len(significant_months)}")

    if len(significant_months) > 0:
        print(
            "Months                      :",
            ", ".join(significant_months["month_id"].astype(str))
        )

    print(f"Primary Driver               : {driver}")

    print("\nRecommendation")
    print("-" * 70)
    print(recommendation)

    print("=" * 70)

    return {
    "Trial Store": trial_store,
    "Control Store": control_store,
    "Sales Uplift (%)": round(sales_uplift, 2),
    "Customer Uplift (%)": round(customer_uplift, 2),
    "Transaction Uplift (%)": round(txn_uplift, 2),
   # "Significant Months": len(significant_months),
    "Insights": driver,
    "Recommendation": recommendation
}


# In[51]:


summary = pd.DataFrame([
    final_summary(77),
    final_summary(86),
    final_summary(88)
])

summary

summary.to_csv(
    "Trial_Store_Summary.csv",
    index=False
)


# In[ ]:




