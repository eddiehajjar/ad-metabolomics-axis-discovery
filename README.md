Patient-Specific Lipidomic Response Axes to Dupilumab

This project analyzes paired, untargeted lipidomics data from patients with atopic dermatitis (AD) treated with dupilumab. The goal is to understand how patients respond to treatment at the metabolic level — not by forcing patients into predefined subtypes, but by learning the dominant treatment response directly from the data.

Instead of asking “Which group does this patient belong to?”, the core question is:
Do patients share a common metabolic response to treatment, and if so, how strongly does each patient express it?


Data Overview
	•	Study: NIH Metabolomics Workbench (ST002302)
	•	Samples: 33 patients with paired serum samples (Before / After dupilumab) + QC samples
	•	Assays: Untargeted lipidomics (HILIC chromatography, positive and negative ionization modes)

Each sample contains hundreds of lipid features, represented by mass-to-charge ratio and retention time (e.g., m/z_rt), with an intensity value per patient. These intensities reflect relative lipid abundance.


Analysis Approach (High-Level)

1. Clean, align, and quality-filter lipidomics data

Raw lipid feature tables are parsed, aligned across samples, and filtered using QC variability to remove unstable signals. Positive and negative ionization modes are processed independently.

2. Compute paired, within-patient changes

For each patient, we calculate how their lipid profile changes from Before to After treatment. This isolates treatment effects from baseline differences between patients.

3. Learn a shared treatment response axis

We compute the average paired change across patients, defining a single direction in high-dimensional lipid space that represents the dominant metabolic response to dupilumab.

Each sample is then projected onto this axis, yielding a scalar axis score:
	•	Higher score → further along the treatment response
	•	Lower score → closer to baseline state

This converts thousands of lipid measurements into an interpretable, patient-specific response coordinate.


Key Findings

A shared metabolic trajectory

Patients move in a highly consistent direction following treatment. Differences between patients are driven primarily by how far they move along the axis — not by different response directions.

Strong POS–NEG concordance

Treatment response magnitude is strongly correlated between positive and negative ionization modes at the per-patient level, indicating the signal reflects true biology rather than technical artifacts.

Baseline state predicts response

Patients’ pre-treatment position along the axis is significantly correlated with how strongly they respond to dupilumab. This suggests baseline metabolism contains predictive information about treatment response.

Interpretable biochemical drivers

Features with the highest axis weights overlap with lipids that show strong paired differential expression, linking the response axis to biologically meaningful lipid pathways implicated in inflammation and AD.

Predictive modeling (exploratory)

Regularized regression models trained on baseline lipidomics show modest but significant ability to predict response magnitude under leave-one-subject-out evaluation — especially in negative ion mode.


Why This Matters

This project demonstrates a general framework for turning high-dimensional, untargeted metabolomics into continuous, patient-specific drug response phenotypes. Rather than labeling patients upfront, the structure of response is learned directly from the data.

The same approach can be applied to:
	•	Other drugs or diseases
	•	Longitudinal metabolomics or proteomics
	•	Precision medicine settings where response exists on a spectrum
