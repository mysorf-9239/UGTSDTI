"""
Demo: Build DD/PP similarity graphs from scratch.

Usage:
    conda run -n ugtsdti python examples/demo_graph_builder.py
"""
from ugtsdti.data.graph_builder import build_drug_drug_graph, build_protein_protein_graph

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    "c1ccc2ccccc2c1",           # Naphthalene
    "OC(=O)c1ccccc1",           # Benzoic acid
]

FASTA = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGEDEDTLSLQHGGTQNLHISRQLEGQHI",
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY",
]

if __name__ == "__main__":
    print("Building Drug-Drug graph...")
    dd = build_drug_drug_graph(SMILES, top_k=2)
    print(f"  nodes={dd.num_nodes}, edges={dd.edge_index.shape[1]}, feat={dd.x.shape}")

    print("Building Protein-Protein graph...")
    pp = build_protein_protein_graph(FASTA, top_k=1)
    print(f"  nodes={pp.num_nodes}, edges={pp.edge_index.shape[1]}, feat={pp.x.shape}")

    print("Done.")
