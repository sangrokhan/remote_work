from pyvis.network import Network
import os
import networkx as nx

def visualize_graph(G, output_path):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
    
    for node in G.nodes():
        net.add_node(node, label=node, title=node, color="#97c2fc")
        
    for source, target, data in G.edges(data=True):
        weight = data['weight']
        # 색상: 양의 상관관계는 초록색 계열, 음의 상관관계는 빨간색 계열
        color = "#00ff00" if weight > 0 else "#ff0000"
        net.add_edge(source, target, value=abs(weight) * 10, title=f"Corr: {weight:.2f}", color=color)
        
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      }
    }
    """)
    
    net.save_graph(output_path)
    print(f"Interactive graph saved at {output_path}")

if __name__ == "__main__":
    from graph_engine import create_correlation_graph
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "kpi_samples.csv")
    output_path = os.path.join(base_dir, "outputs", "interactive_graph.html")
    
    G = create_correlation_graph(data_path)
    visualize_graph(G, output_path)
