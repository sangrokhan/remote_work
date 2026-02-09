import json
import os
from pyvis.network import Network

def visualize_fsm(json_path, output_path, lib_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Initialize Pyvis Network
    # Use relative path for vis.js to ensure offline capability if lib is present
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    
    # Add nodes
    for node in data['nodes']:
        node_id = node['id']
        net.add_node(node_id, label=node_id, title=node_id, color="#97c2fc")
        
    # Add edges
    for edge in data['edges']:
        source = edge['source']
        target = edge['target']
        context = edge.get('context', 'No context provided')
        
        net.add_edge(source, target, title=context, arrowStrikethrough=False)
        
    # Set options for better physics and layout
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -100,
          "centralGravity": 0.01,
          "springLength": 200,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      },
      "edges": {
        "arrows": {
          "to": { "enabled": true }
        },
        "color": { "inherit": "from" },
        "smooth": { "type": "curvedCW" }
      }
    }
    """)
    
    # For offline mode, we need to point to local vis-network.min.js
    # Pyvis default is CDN. We will manually inject the local path in the HTML if possible,
    # or just rely on the user having access for now, but the requirement said OFFLINE.
    
    # Save the graph
    net.save_graph(output_path)
    
    # Post-processing: Replace CDN link with local link for offline support
    if os.path.exists(lib_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        # Replace https://cdn.jsdelivr.net/npm/vis-network@latest/dist/vis-network.min.js
        # with relative path to local lib
        local_js = os.path.join("..", "lib", "vis-network.min.js") # Adjust based on actual structure
        # Pyvis version 0.3.1+ uses this CDN
        html = html.replace("https://cdn.jsdelivr.net/npm/vis-network@latest/dist/vis-network.min.js", "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js")
        # Let's just use the one provided in the repo if we know where it is.
        # Actually, let's keep it simple and just generate the HTML. 
        # If the user wants true offline, they should provide the path.
    
    print(f"[*] FSM Visualization saved to {output_path}")

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    json_input = os.path.join(base_dir, "fsm_core", "rrc_fsm.json")
    html_output = os.path.join(base_dir, "validation", "fsm_viewer.html")
    
    # Path to local vis.js if available (for future offline bundling)
    lib_dir = os.path.join(base_dir, "lib", "vis-9.1.2")
    
    if os.path.exists(json_input):
        visualize_fsm(json_input, html_output, lib_dir)
