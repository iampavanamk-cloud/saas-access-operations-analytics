import pandas as pd
from sklearn.ensemble import IsolationForest
df = pd.read_csv("synthetic_auth_events_180000.csv")
df.columns = df.columns.str.strip()
features = [
    "user_role",
    "auth_protocol",
    "logon_type",
    "auth_result",
    "is_domain_controller_target",
]
X = df[features]
X_encoded = pd.get_dummies(X, drop_first=True)
model = IsolationForest(contamination=0.05, random_state=42)
df["anomaly_score"] = model.fit_predict(X_encoded)
df["is_predicted_anomaly"] = df["anomaly_score"] == -1
print("\n--- Updated Model Evaluation ---")
print(
    pd.crosstab(
        df["is_malicious"],
        df["is_predicted_anomaly"],
        rownames=["Actual Malicious"],
        colnames=["Predicted Anomaly"],
    )
)
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
role_result = pd.crosstab(df['user_role'], df['auth_result'], normalize='index') * 100
role_result.plot(kind='bar', stacked=True, ax=axes[0, 0], colormap='viridis')
axes[0, 0].set_title('Authentication Success Rate by User Role', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('User Role', fontsize=10)
axes[0, 0].set_ylabel('Percentage (%)', fontsize=10)
axes[0, 0].legend(title='Auth Result')
axes[0, 0].tick_params(axis='x', rotation=0)
top_subnets = df['src_subnet'].value_counts().head(5)
sns.barplot(x=top_subnets.values, y=top_subnets.index, ax=axes[0, 1], palette='crest')
axes[0, 1].set_title('Top 5 Source Subnets by Event Volume', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Total Authentication Events', fontsize=10)
axes[0, 1].set_ylabel('Source Subnet', fontsize=10)
protocol_counts = df['auth_protocol'].value_counts()
axes[1, 0].pie(protocol_counts, labels=protocol_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66c2a5', '#fc8d62'])
axes[1, 0].set_title('Authentication Protocol Distribution', fontsize=12, fontweight='bold')
fail_df = df[df['auth_result'] == 'Failure']
fail_by_subnet = fail_df['src_subnet'].value_counts().head(5)
sns.barplot(x=fail_by_subnet.values, y=fail_by_subnet.index, ax=axes[1, 1], palette='magma')
axes[1, 1].set_title('Top 5 Subnets with Highest Authentication Failures', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Total Authentication Failures', fontsize=10)
axes[1, 1].set_ylabel('Source Subnet', fontsize=10)
plt.tight_layout()
plt.savefig('saas_access_analytics_dashboard.png', dpi=300)
plt.show()
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
palette = {"Success": "#2b5c8f", "Failure": "#d95f02"}

df["timestamp"] = pd.to_datetime(df["timestamp"])
hourly_auth = (
    df.groupby([pd.Grouper(key="timestamp", freq="h"), "auth_result"])
    .size()
    .unstack(fill_value=0)
)
hourly_auth_melted = (
    hourly_auth.reset_index()
    .melt(id_vars="timestamp", var_name="auth_result", value_name="count")
)
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
palette = {"Success": "#2b5c8f", "Failure": "#d95f02"}
sns.lineplot(
    data=hourly_auth_melted,
    x="timestamp",
    y="count",
    hue="auth_result",
    palette=palette,
    linewidth=1.8,
    ax=ax,
)
sns.despine(top=True, right=True)
ax.set_title(
    "Hourly Authentication Volume & System Load",
    fontsize=14,
    fontweight="600",
    pad=15,
)
ax.set_xlabel("Timeline (Hourly)", fontsize=11, labelpad=10)
ax.set_ylabel("Authentication Events", fontsize=11, labelpad=10)
ax.legend(title="Auth Status", frameon=True, facecolor="white", edgecolor="none")
ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")

plt.tight_layout()
plt.savefig("clean_saas_auth_trend.png", dpi=300)
plt.show()
