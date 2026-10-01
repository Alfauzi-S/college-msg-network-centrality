import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

FILE_PATH = "data/CollegeMsg.txt"
OUTPUT_DIR = "hasil"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(
    FILE_PATH,
    sep=r"\s+",
    names=["source", "target", "timestamp"]
)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique source nodes:")
print(df["source"].nunique())

print("\nUnique target nodes:")
print(df["target"].nunique())


# ============================================================
# 3. BUILD DIRECTED GRAPH
# ============================================================

G = nx.DiGraph()

for _, row in df.iterrows():
    G.add_edge(
        row["source"],
        row["target"]
    )


number_of_nodes = G.number_of_nodes()
number_of_edges = G.number_of_edges()

print("\n" + "=" * 60)
print("GRAPH INFORMATION")
print("=" * 60)

print(f"\nNumber of nodes : {number_of_nodes}")
print(f"Number of edges : {number_of_edges}")


# ============================================================
# 4. NETWORK CHARACTERISTICS
# ============================================================

density = nx.density(G)

in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

average_in_degree = (
    sum(in_degree.values()) / number_of_nodes
)

average_out_degree = (
    sum(out_degree.values()) / number_of_nodes
)

weak_components = nx.number_weakly_connected_components(G)
strong_components = nx.number_strongly_connected_components(G)

network_statistics = pd.DataFrame({
    "Metric": [
        "Number of interactions",
        "Number of nodes",
        "Number of unique edges",
        "Network density",
        "Average in-degree",
        "Average out-degree",
        "Weakly connected components",
        "Strongly connected components",
        "Duplicate records"
    ],
    "Value": [
        len(df),
        number_of_nodes,
        number_of_edges,
        density,
        average_in_degree,
        average_out_degree,
        weak_components,
        strong_components,
        df.duplicated().sum()
    ]
})

network_statistics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "network_statistics.csv"
    ),
    index=False
)

print("\n" + "=" * 60)
print("NETWORK CHARACTERISTICS")
print("=" * 60)

print(network_statistics.to_string(index=False))


# ============================================================
# 5. DEGREE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DEGREE ANALYSIS")
print("=" * 60)

in_degree = dict(G.in_degree())
out_degree = dict(G.out_degree())

# In-degree centrality
in_degree_centrality = nx.in_degree_centrality(G)

# Out-degree centrality
out_degree_centrality = nx.out_degree_centrality(G)

# Total degree centrality
degree_centrality = nx.degree_centrality(G)


# Create degree dataframe
degree_df = pd.DataFrame({
    "node": list(G.nodes()),
    "in_degree": [
        in_degree[node]
        for node in G.nodes()
    ],
    "out_degree": [
        out_degree[node]
        for node in G.nodes()
    ],
    "in_degree_centrality": [
        in_degree_centrality[node]
        for node in G.nodes()
    ],
    "out_degree_centrality": [
        out_degree_centrality[node]
        for node in G.nodes()
    ],
    "degree_centrality": [
        degree_centrality[node]
        for node in G.nodes()
    ]
})

degree_df["total_degree"] = (
    degree_df["in_degree"] +
    degree_df["out_degree"]
)

degree_df = degree_df[
    [
        "node",
        "in_degree",
        "out_degree",
        "total_degree",
        "in_degree_centrality",
        "out_degree_centrality",
        "degree_centrality"
    ]
]

degree_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "degree_results.csv"
    ),
    index=False
)


# ============================================================
# 6. TOP 10 IN-DEGREE
# ============================================================

top_in_degree = (
    degree_df[
        [
            "node",
            "in_degree",
            "in_degree_centrality"
        ]
    ]
    .sort_values(
        "in_degree",
        ascending=False
    )
    .head(10)
)

top_in_degree.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top10_in_degree.csv"
    ),
    index=False
)


# ============================================================
# 7. TOP 10 OUT-DEGREE
# ============================================================

top_out_degree = (
    degree_df[
        [
            "node",
            "out_degree",
            "out_degree_centrality"
        ]
    ]
    .sort_values(
        "out_degree",
        ascending=False
    )
    .head(10)
)

top_out_degree.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top10_out_degree.csv"
    ),
    index=False
)


# ============================================================
# 8. TOP 10 DEGREE CENTRALITY
# ============================================================

top_degree = (
    degree_df[
        [
            "node",
            "degree_centrality"
        ]
    ]
    .sort_values(
        "degree_centrality",
        ascending=False
    )
    .head(10)
)

top_degree.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top10_degree_centrality.csv"
    ),
    index=False
)


# ============================================================
# 9. DISPLAY DEGREE RESULTS
# ============================================================

print("\nTop 10 In-Degree:")
print(
    top_in_degree.to_string(
        index=False
    )
)

print("\nTop 10 Out-Degree:")
print(
    top_out_degree.to_string(
        index=False
    )
)

print("\nTop 10 Degree Centrality:")
print(
    top_degree.to_string(
        index=False
    )
)


# ============================================================
# 10. BETWEENNESS CENTRALITY
# ============================================================

print("\nCalculating betweenness centrality...")

betweenness_centrality = (
    nx.betweenness_centrality(
        G,
        normalized=True
    )
)


# ============================================================
# 11. CLOSENESS CENTRALITY
# ============================================================

print("Calculating closeness centrality...")

# Inward closeness:
# shortest-path distance from other nodes
# toward the target node.
inward_closeness = (
    nx.closeness_centrality(G)
)

# Outward closeness:
# shortest-path distance from the node
# toward other nodes.
outward_closeness = (
    nx.closeness_centrality(
        G.reverse()
    )
)


# ============================================================
# 12. CENTRALITY DATAFRAME
# ============================================================

centrality_df = degree_df.copy()

centrality_df["betweenness_centrality"] = [
    betweenness_centrality[node]
    for node in centrality_df["node"]
]

centrality_df["inward_closeness"] = [
    inward_closeness[node]
    for node in centrality_df["node"]
]

centrality_df["outward_closeness"] = [
    outward_closeness[node]
    for node in centrality_df["node"]
]

centrality_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "centrality_results.csv"
    ),
    index=False
)


# ============================================================
# 13. TOP 10 BETWEENNESS
# ============================================================

top_betweenness = (
    centrality_df[
        [
            "node",
            "betweenness_centrality"
        ]
    ]
    .sort_values(
        "betweenness_centrality",
        ascending=False
    )
    .head(10)
)

top_betweenness.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top10_betweenness.csv"
    ),
    index=False
)


# ============================================================
# 14. TOP 10 INWARD CLOSENESS
# ============================================================

top_inward_closeness = (
    centrality_df[
        [
            "node",
            "inward_closeness"
        ]
    ]
    .sort_values(
        "inward_closeness",
        ascending=False
    )
    .head(10)
)

top_inward_closeness.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top10_inward_closeness.csv"
    ),
    index=False
)


# ============================================================
# 15. TOP 10 OUTWARD CLOSENESS
# ============================================================

top_outward_closeness = (
    centrality_df[
        [
            "node",
            "outward_closeness"
        ]
    ]
    .sort_values(
        "outward_closeness",
        ascending=False
    )
    .head(10)
)

top_outward_closeness.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "top10_outward_closeness.csv"
    ),
    index=False
)


# ============================================================
# 16. DISPLAY CENTRALITY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TOP 10 BETWEENNESS CENTRALITY")
print("=" * 60)

print(
    top_betweenness.to_string(
        index=False
    )
)


print("\n" + "=" * 60)
print("TOP 10 INWARD CLOSENESS")
print("=" * 60)

print(
    top_inward_closeness.to_string(
        index=False
    )
)


print("\n" + "=" * 60)
print("TOP 10 OUTWARD CLOSENESS")
print("=" * 60)

print(
    top_outward_closeness.to_string(
        index=False
    )
)

# ============================================================
# 17. DEGREE DISTRIBUTION
# ============================================================

print("\nGenerating degree distribution graph...")

plt.figure(figsize=(10, 6))

plt.hist(
    degree_df["total_degree"],
    bins=50
)

plt.title(
    "Degree Distribution of CollegeMsg Network"
)

plt.xlabel("Total Degree")
plt.ylabel("Number of Nodes")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "degree_distribution.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 18. TOP 10 DEGREE CENTRALITY GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    top_degree["node"].astype(str),
    top_degree["degree_centrality"]
)

plt.title(
    "Top 10 Degree Centrality"
)

plt.xlabel("Node")
plt.ylabel("Degree Centrality")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "top10_degree_centrality.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 19. TOP 10 BETWEENNESS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    top_betweenness["node"].astype(str),
    top_betweenness["betweenness_centrality"]
)

plt.title(
    "Top 10 Betweenness Centrality"
)

plt.xlabel("Node")
plt.ylabel("Betweenness Centrality")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "top10_betweenness.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 20. TOP 10 INWARD CLOSENESS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    top_inward_closeness["node"].astype(str),
    top_inward_closeness["inward_closeness"]
)

plt.title(
    "Top 10 Inward Closeness Centrality"
)

plt.xlabel("Node")
plt.ylabel("Inward Closeness Centrality")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "top10_inward_closeness.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 21. TOP 10 OUTWARD CLOSENESS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    top_outward_closeness["node"].astype(str),
    top_outward_closeness["outward_closeness"]
)

plt.title(
    "Top 10 Outward Closeness Centrality"
)

plt.xlabel("Node")
plt.ylabel("Outward Closeness Centrality")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "top10_outward_closeness.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 22. CENTRALITY COMPARISON
# ============================================================

centrality_comparison = centrality_df[
    [
        "node",
        "degree_centrality",
        "betweenness_centrality",
        "inward_closeness",
        "outward_closeness"
    ]
].copy()

centrality_comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "centrality_comparison.csv"
    ),
    index=False
)


# ============================================================
# 23. ANALYSIS COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)

print(
    f"\nAll analysis results have been saved to: "
    f"{OUTPUT_DIR}/"
)

print("\nGenerated files:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):
    print(f"- {filename}")