# DS3000 – Predicting Hospital Patient Stay Length

## Overview

Nurses are massively overworked, and a lot of their time gets wasted on planning and admin tasks instead of direct patient care. In Canada, about 94% of nurses report some level of burnout. This project uses non-medical (administrative) hospital data to predict key operational metrics like patient length of stay and potential bed usage. The goal is to build a model that can plug into a larger tool to help nurses plan, prioritize patients, and reduce some of that planning burden. :contentReference[oaicite:0]{index=0}  

Concretely, we:

- Use a Kaggle dataset with hospital admission information and administrative metadata.
- Predict the **“Stay”** field (length of stay category) using non-medical features.
- Explore models that can handle non-linear relationships and imbalanced data.

---

## Problem Statement

Nurses face intense workloads and burnout, with a lot of non-nursing tasks added on top of clinical care. Our project focuses on:

- Using **administrative data** (hospital semantics, room availability, illness type/severity, etc.) to predict:
  - Patient **length of stay**
  - Potential **bed occupancy patterns**
- Using these predictions as a building block for:
  - Simplified summaries of upcoming patient loads
  - Automated or semi-automated scheduling and prioritization tools for nurses :contentReference[oaicite:1]{index=1}  

---

## Dataset

We use a Kaggle dataset that includes four CSV files: :contentReference[oaicite:2]{index=2}  

- `train.csv`  
  - ~318,000 rows  
  - Contains the **target column**: `Stay` (length of stay)
  - Used for training and validation
- `test.csv`  
  - Does **not** include `Stay`
  - Intended for Kaggle submissions and will only be used later once we have a final model
- `sample_submission.csv`
  - Just a sample table, not relevant to this project 
- `dictionary.csv`  
  - Describes each feature and its meaning, referenced in our notebooks

Key findings from Phase 1:

- The data is **imbalanced** – many features have skewed distributions across categories.
- The **correlation matrix** shows very weak linear relationships with `Stay`:
  - Only one feature has ~0.53 correlation with `Stay`
  - The next highest is ~0.18  
  → Pure linear models are likely a bad fit.
- A temporary **Random Forest** model shows a relatively spread-out feature importance, with the top feature around 14% importance. :contentReference[oaicite:3]{index=3}

---

## Contributors
Group 15, consists of:

- Kwaku Asare
- Tejanvesh Gangavarapu
- Obaid Mohiuddin
- Zain Syed

---

