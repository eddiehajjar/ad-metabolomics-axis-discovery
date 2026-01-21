Patient-Specific Lipidomic Response Axes to Dupilumab

This project analyzes paired, untargeted lipidomics from patients with atopic dermatitis (AD) treated with dupilumab to understand how drug response emerges in high-dimensional metabolic space. Rather than classifying patients into predefined subtypes, we discover a shared treatment response axis directly from the data and quantify how individual patients move along it.

Using paired before/after serum samples, we compute within-patient lipidomic changes and define a population-level response direction. Each sample is projected onto this axis to obtain a scalar response score, allowing direct comparison of response magnitude across patients. This reveals that dupilumab induces a coherent metabolic program, with patients differing primarily in how far they move along a common trajectory rather than in the direction of response.

The analysis is performed independently in positive and negative ionization modes and shows strong per-patient concordance across modes, confirming that the signal reflects biology rather than technical artifacts. Baseline axis position is significantly correlated with response magnitude, indicating that pre-treatment metabolic state predicts downstream drug response.

We further connect axis weights to differential lipid features, identifying interpretable biochemical drivers consistent with inflammatory and lipid-signaling pathways implicated in AD. Exploratory regularized models using baseline lipidomics demonstrate modest but significant ability to predict response magnitude under leave-one-subject-out evaluation.

Overall, this project demonstrates a generalizable framework for extracting continuous, patient-specific drug response phenotypes from untargeted metabolomics—aligned with AI-driven discovery approaches that learn biological structure first and apply labels second.

Data: NIH Metabolomics Workbench (ST002302)
Samples: 33 patients, paired before/after dupilumab + QC
Assays: Untargeted lipidomics (HILIC, POS/NEG ESI)

conda env create -f environment.yaml
conda activate ad-metabolomics
bash scripts/run_all.sh
