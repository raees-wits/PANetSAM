from graphviz import Digraph
import os

def create_high_level_diagram():
    dot = Digraph(comment='ProtoNetSAM Architecture')
    dot.attr(rankdir='TB')
    
    # Define node styles
    dot.attr('node', shape='box', 
             style='filled,rounded', 
             fillcolor='lightblue', 
             fontname='Arial',
             margin='0.3,0.1')
    
    # Input nodes
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Inputs')
        c.attr(style='rounded')
        c.node('input_support', 'Support Images\n& Masks')
        c.node('input_query', 'Query Images')
    
    # SAM base nodes
    with dot.subgraph(name='cluster_sam') as c:
        c.attr(label='SAM Components')
        c.attr(style='rounded')
        c.node('sam_encoder', 'SAM Image Encoder\n(frozen)')
        c.node('sam_prompt', 'SAM Prompt Encoder')
        c.node('sam_decoder', 'SAM Mask Decoder')
    
    # Prototype learning nodes
    with dot.subgraph(name='cluster_proto') as c:
        c.attr(label='Prototype Learning')
        c.attr(style='rounded')
        c.node('proto_adaptor', 'Prototype Adaptor')
        c.node('proto_compute', 'Prototype Computation')
        c.node('sim_maps', 'Similarity Maps')
        c.node('box_gen', 'Box Generation')
    
    # Output nodes
    dot.node('decoder', 'Final Decoder')
    dot.node('output', 'Output Masks')
    
    # Add edges
    # Main flow
    dot.edge('input_support', 'sam_encoder')
    dot.edge('input_query', 'sam_encoder')
    dot.edge('sam_encoder', 'proto_adaptor')
    dot.edge('proto_adaptor', 'proto_compute')
    dot.edge('proto_compute', 'sim_maps')
    
    # Prompt flow
    dot.edge('sim_maps', 'box_gen', 'Generate Prompt Box')
    dot.edge('box_gen', 'sam_prompt')
    
    # SAM processing
    dot.edge('sam_prompt', 'sam_decoder')
    dot.edge('sam_encoder', 'sam_decoder')
    
    # Final processing
    dot.edge('sim_maps', 'decoder')
    dot.edge('sam_decoder', 'decoder')
    dot.edge('decoder', 'output')
    
    # Add legend for prompt types
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Prompt Types', style='rounded')
        c.node('legend1', 'GT Boxes (when available)', shape='note')
        c.node('legend2', 'Auto-generated from\nsimilarity maps', shape='note')
    
    # Save the diagram
    os.makedirs('model_viz', exist_ok=True)
    dot.render('model_viz/protonetsam_high_level', format='svg', cleanup=True)

if __name__ == "__main__":
    create_high_level_diagram()