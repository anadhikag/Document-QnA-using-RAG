import plotly.graph_objects as go
import plotly.express as px
import json

# Data with improved layout (top-to-bottom flow)
data = {
  "components": [
    {"layer": "Input", "name": "PDF Upload", "x": 0.5, "y": 7},
    {"layer": "Input", "name": "HTML/MD Upload", "x": 1.5, "y": 7},
    {"layer": "Input", "name": "URL Input", "x": 2.5, "y": 7},
    {"layer": "Input", "name": "User Query", "x": 4, "y": 7},
    {"layer": "Processing", "name": "Document Ingestion", "x": 1.5, "y": 6},
    {"layer": "Processing", "name": "Overlap Chunker", "x": 1.5, "y": 5},
    {"layer": "Processing", "name": "Sentence-Transformers", "x": 0.8, "y": 4},
    {"layer": "Processing", "name": "OpenAI Embeddings", "x": 2.2, "y": 4},
    {"layer": "Storage", "name": "ChromaDB Vector Store", "x": 1.5, "y": 3},
    {"layer": "Retrieval", "name": "Top-k Search", "x": 2.5, "y": 2},
    {"layer": "Generation", "name": "Ollama/phi3", "x": 1.8, "y": 1},
    {"layer": "Generation", "name": "OpenAI GPT", "x": 3.2, "y": 1},
    {"layer": "Output", "name": "Streamlit UI", "x": 2.5, "y": 0.2},
    {"layer": "Output", "name": "Citations & Export", "x": 3.8, "y": 0.2}
  ],
  "flows": [
    {"from": "PDF Upload", "to": "Document Ingestion"},
    {"from": "HTML/MD Upload", "to": "Document Ingestion"},
    {"from": "URL Input", "to": "Document Ingestion"},
    {"from": "Document Ingestion", "to": "Overlap Chunker"},
    {"from": "Overlap Chunker", "to": "Sentence-Transformers"},
    {"from": "Overlap Chunker", "to": "OpenAI Embeddings"},
    {"from": "Sentence-Transformers", "to": "ChromaDB Vector Store"},
    {"from": "OpenAI Embeddings", "to": "ChromaDB Vector Store"},
    {"from": "ChromaDB Vector Store", "to": "Top-k Search"},
    {"from": "User Query", "to": "Top-k Search"},
    {"from": "Top-k Search", "to": "Ollama/phi3"},
    {"from": "Top-k Search", "to": "OpenAI GPT"},
    {"from": "Ollama/phi3", "to": "Streamlit UI"},
    {"from": "OpenAI GPT", "to": "Streamlit UI"},
    {"from": "Streamlit UI", "to": "Citations & Export"}
  ]
}

# Brand colors
colors = ["#1FB8CD", "#DB4545", "#2E8B57", "#5D878F", "#D2BA4C", "#B4413C"]
layer_colors = {
    "Input": colors[0],
    "Processing": colors[1], 
    "Storage": colors[2],
    "Retrieval": colors[3],
    "Generation": colors[4],
    "Output": colors[5]
}

# Better abbreviations within 15 char limit
name_mapping = {
    "Document Ingestion": "Doc Ingestion",
    "Overlap Chunker": "Chunker",
    "Sentence-Transformers": "SentTransform", 
    "OpenAI Embeddings": "OpenAI Embed",
    "ChromaDB Vector Store": "Vector Store",
    "Top-k Search": "Semantic Search",
    "Ollama/phi3": "Ollama/phi3",
    "OpenAI GPT": "OpenAI GPT",
    "Streamlit UI": "Streamlit UI",
    "Citations & Export": "Cite & Export",
    "PDF Upload": "PDF Upload",
    "HTML/MD Upload": "HTML/MD",
    "URL Input": "URL Input",
    "User Query": "User Query"
}

# Create coordinate lookup
coord_lookup = {}
for comp in data["components"]:
    coord_lookup[comp["name"]] = (comp["x"], comp["y"])

# Create figure
fig = go.Figure()

# Add flow lines with proper arrows
for flow in data["flows"]:
    from_coords = coord_lookup[flow["from"]]
    to_coords = coord_lookup[flow["to"]]
    
    # Calculate arrow position (stop before reaching the box)
    dx = to_coords[0] - from_coords[0]
    dy = to_coords[1] - from_coords[1]
    length = (dx**2 + dy**2)**0.5
    
    if length > 0:
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Stop line before box (adjust for box size)
        end_x = to_coords[0] - dx_norm * 0.25
        end_y = to_coords[1] - dy_norm * 0.15
        
        # Draw main line
        fig.add_shape(
            type="line",
            x0=from_coords[0], y0=from_coords[1] - 0.15,
            x1=end_x, y1=end_y,
            line=dict(color="#666666", width=2),
            opacity=0.7
        )
        
        # Add arrowhead using annotation
        fig.add_annotation(
            x=end_x,
            y=end_y,
            ax=from_coords[0],
            ay=from_coords[1] - 0.15,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#666666",
            opacity=0.7
        )

# Add components by layer with better sizing
for layer in ["Input", "Processing", "Storage", "Retrieval", "Generation", "Output"]:
    layer_components = [c for c in data["components"] if c["layer"] == layer]
    
    x_coords = [c["x"] for c in layer_components]
    y_coords = [c["y"] for c in layer_components]
    names = [name_mapping.get(c["name"], c["name"]) for c in layer_components]
    
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+text',
        text=names,
        textposition='middle center',
        textfont=dict(size=9, color='white', family='Arial'),
        marker=dict(
            size=100,
            color=layer_colors[layer],
            symbol='square',
            line=dict(width=2, color='white')
        ),
        name=layer,
        hovertemplate='%{text}<br>Layer: ' + layer + '<extra></extra>'
    ))

# Add grouping boxes for modular interfaces
# Embedding providers group
fig.add_shape(
    type="rect",
    x0=0.4, y0=3.7, x1=2.6, y1=4.3,
    line=dict(color="#999999", width=2, dash="dash"),
    fillcolor="rgba(0,0,0,0)"
)
fig.add_annotation(
    x=1.5, y=4.4,
    text="Embed Providers",
    showarrow=False,
    font=dict(size=10, color="#666666")
)

# LLM providers group  
fig.add_shape(
    type="rect",
    x0=1.4, y0=0.7, x1=3.6, y1=1.3,
    line=dict(color="#999999", width=2, dash="dash"),
    fillcolor="rgba(0,0,0,0)"
)
fig.add_annotation(
    x=2.5, y=1.4,
    text="LLM Providers",
    showarrow=False,
    font=dict(size=10, color="#666666")
)

# Update layout
fig.update_layout(
    title="Document QnA RAG Architecture",
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    xaxis=dict(
        showgrid=False, 
        showticklabels=False, 
        zeroline=False,
        range=[-0.2, 4.5]
    ),
    yaxis=dict(
        showgrid=False, 
        showticklabels=False, 
        zeroline=False,
        range=[-0.5, 7.8]
    ),
    plot_bgcolor='white',
    hovermode='closest'
)

fig.update_traces(cliponaxis=False)

# Save the chart
fig.write_image("rag_architecture.png", width=1000, height=800)
fig.show()