"""
Demo: Run a forward pass through GCNTeacher with toy data.

Usage:
    conda run -n ugtsdti python examples/demo_gcn_teacher.py
"""
import torch
from ugtsdti.data.graph_builder import build_drug_drug_graph, build_protein_protein_graph
from ugtsdti.models.teacher.gcn_teacher import GCNTeacher

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "c1ccc2ccccc2c1",
    "OC(=O)c1ccccc1",
]
FASTA = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGEDEDTLSLQHGGTQNLHISRQLEGQHI",
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY",
]

if __name__ == "__main__":
    dd = build_drug_drug_graph(SMILES, top_k=2)
    pp = build_protein_protein_graph(FASTA, top_k=1)

    model = GCNTeacher(drug_feat_dim=2048, protein_feat_dim=8000, hidden_dim=64, num_layers=2)
    model.set_graphs(dd, pp)
    model.eval()

    # Batch: 3 pairs
    batch = {
        "drug_index": torch.tensor([0, 1, 2]),
        "target_index": torch.tensor([0, 1, 0]),
    }
    out = model(batch)
    print(f"logits shape: {out['logits'].shape}")   # (3,)
    print(f"logits: {out['logits'].detach()}")
