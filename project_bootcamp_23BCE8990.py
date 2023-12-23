import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('heart_failure_clinical_records.csv')


print("First few rows of the dataset:")
print(df.head())
print("\nInformation about columns and data types:")
print(df.info())

print("\nViewing specific columns (e.g., 'Age', 'Sex', 'Death Event'):")
print(df[['age', 'sex', 'DEATH_EVENT']].head())

columns_to_drop = ['anaemia', 'diabetes']
df.drop(columns=columns_to_drop,inplace=True)
print("\nDataset after dropping 'Anaemia' and 'Diabetes' columns:")
print(df.head())

filtered_df = df[df['age'] > 70]
print("\nFiltered rows where Age is greater than 70:")
print(filtered_df.head())

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

axes[0].hist(df['age'], bins=20, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram of Age')
axes[0].grid(True)

ds = df.groupby(['smoking', 'DEATH_EVENT']).size().unstack()
percentages = ds.div(ds.sum(axis=1), axis=0) * 100

bars = ds.plot(kind='bar', stacked=True, color=['green', 'red'], ax=axes[1])
axes[1].set_xlabel('Smoking')
axes[1].set_ylabel('Count')
axes[1].set_title('Count of Death Events by Smoking Status')
axes[1].legend(title='Death Event', labels=['Survived', 'Died'])
axes[1].set_xticklabels(ds.index, rotation=0)

for bar in bars.patches:
    width = bar.get_width()
    height = bar.get_height()
    x, y = bar.get_xy()
    axes[1].text(x + width / 2, y + height / 2, f'{height:.1f}%', ha='center', va='center')

plt.tight_layout()
plt.show()
