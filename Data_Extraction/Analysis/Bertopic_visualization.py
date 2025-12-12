import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# 1. Load Files
# -------------------------------

topic_file = "Data_Extraction/Analysis/Output/BERTOPIC_results_stopwords_v3_mpnet/topic_keywords.csv"          # the big file with Topic, Count, Name, Representation, etc.
review_file = "Data_Extraction/Analysis/Output/BERTOPIC_results_stopwords_v3_mpnet/reviews_with_topics.csv"        # your sample review file (review, topic, etc.)

df_reviews = pd.read_csv(
    review_file,
    engine="python",
    quotechar='"',
    escapechar='\\',
    on_bad_lines='skip'   # or 'warn' if you want to see which lines failed
)

df_topics = pd.read_csv(
    topic_file,
    engine="python",
    quotechar='"',
    escapechar='\\',
    on_bad_lines='skip'
)


print("Loaded:")
print(df_topics.head())
print(df_reviews.head())

# -------------------------------
# 2. Clean Representation Column
# -------------------------------
import ast

def parse_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df_topics["Keywords"] = df_topics["Representation"].apply(parse_list)

# -------------------------------
# 3. Visualization: Bar Chart of Topic Counts
# -------------------------------

fig = px.bar(
    df_topics, 
    x="Topic", 
    y="Count",
    hover_data=["Name"],
    title="Topic Frequency Distribution",
    labels={"Count": "Number of Reviews", "Topic": "Topic ID"},
)

fig.update_layout(xaxis={'type': 'category'})
fig.show()

# -------------------------------
# 4. Treemap of Topics
# -------------------------------

fig = px.treemap(
    df_topics, 
    path=["Topic", "Name"], 
    values="Count",
    title="Topic Treemap (Sized by Count)",
)
fig.show()

# -------------------------------
# 5. Keyword Visualization Per Topic (Top N Words)
# -------------------------------

rows = []
for _, row in df_topics.iterrows():
    topic = row["Topic"]
    name = row["Name"]
    for word in row["Keywords"][:10]:  # top 10 words only
        rows.append({"Topic": topic, "Name": name, "Keyword": word})

df_kw = pd.DataFrame(rows)

fig = px.bar(
    df_kw, 
    x="Keyword", 
    color="Topic",
    title="Top Keywords for Each Topic",
)
fig.show()

# -------------------------------
# 6. Sunburst (Topic Hierarchy)
# -------------------------------

fig = px.sunburst(
    df_topics,
    path=["Topic", "Name"],
    values="Count",
    title="Sunburst Visualization of Topics"
)
fig.show()

# -------------------------------
# 7. (Optional) Scatter Visualization (No Embedding Provided)
# Create synthetic positions for now
# -------------------------------

import numpy as np

df_topics["x"] = np.random.randn(len(df_topics))
df_topics["y"] = np.random.randn(len(df_topics))

fig = px.scatter(
    df_topics,
    x="x",
    y="y",
    size=[40]*len(df_topics),   # make all points larger
    color="Topic",
    hover_data=["Name", "Representation"],
    title="Topic Scatter Plot (UMAP Projection)"
)
fig.show()

print("All visualizations generated successfully.")
