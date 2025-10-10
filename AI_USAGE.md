# AI Usage Log

This project was built with a human-in-the-loop workflow. I used AI assistance to speed up development, but every material artifact was reviewed and tested manually.

## What AI Helped With

- Generated starter code for the Streamlit layout, feature engineering helpers, and portions of the modeling pipeline.
- Suggested performance optimisations (Streamlit caching, precomputing importance artifacts) that I adapted for this repository.
- Drafted explanatory copy in the app and README, which I edited for tone and accuracy.

## What I Verified or Fixed

- Regenerated data splits, models, and metrics locally before deployment to confirm results and ensure thresholds met the recall ≥ 60% requirement.
- Validated business insights (contract length, billing method, support add-ons, CLV segments) against the processed training set before including them in the app and README.
- Hardened the code paths for SHAP fallbacks, caching, and artifact regeneration after testing failure scenarios where optional dependencies (e.g., `shap`) are absent.

## Prompts That Mattered

- “Optimise the Streamlit app so it uses cached assets and avoids recomputation on each rerun.”
- “Add actionable churn-retention insights tied to CLV quartiles and support add-ons.”
- “Draft an AI usage statement that is transparent about assistance but emphasises manual validation.”

I reviewed all AI-generated outputs, corrected mistakes, and confirmed the final behaviour aligns with the grading rubric (business framing, model metrics, CLV insights, and deployment requirements).
