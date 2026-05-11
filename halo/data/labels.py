"""Single source of truth for label columns and amino-acid properties."""

LABEL_COLUMNS = [
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic reticulum",
    "Lysosome/Vacuole",
    "Golgi apparatus",
    "Peroxisome",
]
NUM_LABELS = len(LABEL_COLUMNS)

AMINO_ACID_PROPERTIES: dict[str, dict[str, float]] = {
    "ALA": {"hydrophobicity": 1.8, "charge": 0, "polarity": 8.1, "weight": 89.1},
    "ARG": {"hydrophobicity": -4.5, "charge": 1, "polarity": 10.5, "weight": 174.2},
    "ASN": {"hydrophobicity": -3.5, "charge": 0, "polarity": 11.6, "weight": 132.1},
    "ASP": {"hydrophobicity": -3.5, "charge": -1, "polarity": 13.0, "weight": 133.1},
    "CYS": {"hydrophobicity": 2.5, "charge": 0, "polarity": 5.5, "weight": 121.2},
    "GLN": {"hydrophobicity": -3.5, "charge": 0, "polarity": 10.5, "weight": 146.2},
    "GLU": {"hydrophobicity": -3.5, "charge": -1, "polarity": 12.3, "weight": 147.1},
    "GLY": {"hydrophobicity": -0.4, "charge": 0, "polarity": 9.0, "weight": 75.1},
    "HIS": {"hydrophobicity": -3.2, "charge": 0.5, "polarity": 10.4, "weight": 155.2},
    "ILE": {"hydrophobicity": 4.5, "charge": 0, "polarity": 5.2, "weight": 131.2},
    "LEU": {"hydrophobicity": 3.8, "charge": 0, "polarity": 4.9, "weight": 131.2},
    "LYS": {"hydrophobicity": -3.9, "charge": 1, "polarity": 11.3, "weight": 146.2},
    "MET": {"hydrophobicity": 1.9, "charge": 0, "polarity": 5.7, "weight": 149.2},
    "PHE": {"hydrophobicity": 2.8, "charge": 0, "polarity": 5.2, "weight": 165.2},
    "PRO": {"hydrophobicity": -1.6, "charge": 0, "polarity": 8.0, "weight": 115.1},
    "SER": {"hydrophobicity": -0.8, "charge": 0, "polarity": 9.2, "weight": 105.1},
    "THR": {"hydrophobicity": -0.7, "charge": 0, "polarity": 8.6, "weight": 119.1},
    "TRP": {"hydrophobicity": -0.9, "charge": 0, "polarity": 5.4, "weight": 204.2},
    "TYR": {"hydrophobicity": -1.3, "charge": 0, "polarity": 6.2, "weight": 181.2},
    "VAL": {"hydrophobicity": 4.2, "charge": 0, "polarity": 5.9, "weight": 117.1},
}
BIO_FEATURE_DIM = 4


def get_biochemical_properties(residue_name: str) -> list[float]:
    if residue_name in AMINO_ACID_PROPERTIES:
        return list(AMINO_ACID_PROPERTIES[residue_name].values())
    return [0.0, 0.0, 0.0, 0.0]
