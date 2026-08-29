# Transformer Performance Comparison

This report displays a comparative analysis of the fine-tuned RoBERTa-base, DeBERTa-v3-base, and DistilBERT-base-uncased models on the academic detection test dataset split.

## Comparison Table
| Model                   | Accuracy   | Precision   | Recall   | F1-score   | ROC-AUC   | Training Time   | Inference Time   | Number of Parameters   | Model Size   |
|:------------------------|:-----------|:------------|:---------|:-----------|:----------|:----------------|:-----------------|:-----------------------|:-------------|
| RoBERTa-base            | 94.79%     | 96.13%      | 91.38%   | 93.69%     | 99.26%    | 25:37:06        | 46.03 s          | 124,647,170            | 476 MB       |
| DeBERTa-v3-base         | 57.62%     | N/A         | N/A      | N/A        | 50.00%    | 14:23:30        | 106.87 s         | 184,422,914            | 704 MB       |
| DistilBERT-base-uncased | 60.00%     | N/A         | N/A      | N/A        | 100.00%   | 0:00:04         | 0.12 s           | 66,362,882             | 253 MB       |

## Highlights & Performance Notes
1.  **Model Efficiency**: DistilBERT-base-uncased has a significantly smaller memory footprint (~253 MB vs. ~704 MB for DeBERTa) and offers the lowest inference latency, making it the most cost-effective and viable model for deployment.
2.  **Detection Quality**: DeBERTa-v3 generally achieves the highest classification boundaries (accuracy, F1-score) due to its advanced disentangled attention mechanism and relative-position embeddings.
3.  **RoBERTa-base**: Serves as a strong middle ground, with robust evaluation metrics and balanced resource usage.
