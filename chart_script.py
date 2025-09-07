import plotly.graph_objects as go
import plotly.express as px
import json

data = {
  "components": [
    # Input Layer
    {"layer": "Input", "name": "PDF Upload", "x": 0.5, "y": 7},
    {"layer": "Input", "name": "HTML/MD Upload", "x": 1.5, "y": 7},
    {"layer": "Input", "name": "URL Input", "x": 2.5, "y": 7},
    {"layer": "Input", "name": "User Query", "x": 4, "y": 7},
    
    # Processing Layer
    {"layer": "Processing", "name": "Document Ingestion", "x": 1.5, "y": 6},
    {"layer": "Processing", "name": "Overlap Chunker", "x": 1.5, "y": 5},
    {"layer": "Processing", "name": "Sentence-Transformers", "x": 1.5, "y": 4}, 
    
    # Storage Layer
    {"layer": "Storage", "name": "ChromaDB Vector Store", "x": 1.5, "y": 3},
    
    # Retrieval Layer
    {"layer": "Retrieval", "name": "Top-k Search", "x": 2.75, "y": 2}, 

    # Generation Layer
    {"layer": "Generation", "name": "Ollama/phi3", "x": 2.75, "y": 1}, 
    
    # Output Layer
    {"layer": "Output", "name": "Streamlit UI", "x": 2.0, "y": 0.2},
    {"layer": "Output", "name": "Citations & Export", "x": 3.5, "y": 0.2}
  ],
  "flows": [
    # Ingestion Flow
    {"from": "PDF Upload", "to": "Document Ingestion"},
    {"from": "HTML/MD Upload", "to": "Document Ingestion"},
    {"from": "URL Input", "to": "Document Ingestion"},
    {"from": "Document Ingestion", "to": "Overlap Chunker"},
    {"from": "Overlap Chunker", "to": "Sentence-Transformers"},
    {"from": "Sentence-Transformers", "to": "ChromaDB Vector Store"},
    
    # Query Flow
    {"from": "ChromaDB Vector Store", "to": "Top-k Search"},
    {"from": "User Query", "to": "Top-k Search"},
    {"from": "Top-k Search", "to": "Ollama/phi3"},
    {"from": "Ollama/phi3", "to": "Streamlit UI"},
    {"from": "Streamlit UI", "to": "Citations & Export"}
  ]
}

colors = ["#1FB8CD", "#DB4545", "#2E8B57", "#5D878F", "#D2BA4C", "#B4413C"]
layer_colors = {
    "Input": colors[0],
    "Processing": colors[1], 
    "Storage": colors[2],
    "Retrieval": colors[3],
    "Generation": colors[4],
    "Output": colors[5]
}

name_mapping = {
    "Document Ingestion": "Doc Ingestion",
    "Overlap Chunker": "Chunker",
    "Sentence-Transformers": "Sentence-Transformers", 
    "ChromaDB Vector Store": "Vector Store",
    "Top-k Search": "Semantic Search",
    "Ollama/phi3": "Ollama/phi3",
    "Streamlit UI": "Streamlit UI",
    "Citations & Export": "Cite & Export",
    "PDF Upload": "PDF Upload",
    "HTML/MD Upload": "HTML/MD",
    "URL Input": "URL Input",
    "User Query": "User Query"
}

coord_lookup = {}
for comp in data["components"]:
    coord_lookup[comp["name"]] = (comp["x"], comp["y"])

fig = go.Figure()

for flow in data["flows"]:
    from_coords = coord_lookup[flow["from"]]
    to_coords = coord_lookup[flow["to"]]
    
    dx = to_coords[0] - from_coords[0]
    dy = to_coords[1] - from_coords[1]
    length = (dx**2 + dy**2)**0.5
    
    if length > 0:
        dx_norm, dy_norm = dx / length, dy / length
        end_x, end_y = to_coords[0] - dx_norm * 0.25, to_coords[1] - dy_norm * 0.15
        
        fig.add_shape(
            type="line", x0=from_coords[0], y0=from_coords[1], x1=end_x, y1=end_y,
            line=dict(color="#666666", width=2), opacity=0.7
        )
        
        fig.add_annotation(
            x=end_x, y=end_y, ax=from_coords[0], ay=from_coords[1],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            arrowcolor="#666666", opacity=0.7
        )

for layer in layer_colors.keys():
    layer_components = [c for c in data["components"] if c["layer"] == layer]
    
    fig.add_trace(go.Scatter(
        x=[c["x"] for c in layer_components],
        y=[c["y"] for c in layer_components],
        mode='markers+text',
        text=[name_mapping.get(c["name"], c["name"]) for c in layer_components],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial, sans-serif'),
        marker=dict(
            size=110, color=layer_colors[layer], symbol='square',
            line=dict(width=2, color='white')
        ),
        name=layer,
        hovertemplate='%{text}<br>Layer: ' + layer + '<extra></extra>'
    ))

fig.update_layout(
    title="Document QnA RAG Architecture",
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.2, 5.0]),
    yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 7.8]),
    plot_bgcolor='white',
    hovermode='closest',
    margin=dict(l=20, r=20, t=80, b=20)
)

fig.update_traces(cliponaxis=False)

fig.write_image("rag_architecture.png", width=1000, height=800, scale=2)
print("Successfully generated updated architecture diagram!")
fig.show()
