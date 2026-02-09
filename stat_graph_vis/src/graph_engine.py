import pandas as pd
import networkx as nx
import os

def create_correlation_graph(csv_path, threshold=0.3):
    df = pd.read_csv(csv_path)
    corr_matrix = df.corr()
    
    G = nx.Graph()
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            weight = corr_matrix.iloc[i, j]
            
            if abs(weight) >= threshold:
                G.add_edge(col1, col2, weight=float(weight))
                
    return G

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "kpi_samples.csv")
    graph = create_correlation_graph(data_path)
    print(f"Nodes: {graph.nodes()}")
    print(f"Edges: {len(graph.edges())}")
