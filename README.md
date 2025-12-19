<p align="center"><b>🛡️ Implementation of AI-Based Cognitive Firewall in Banking Sector</b></p>
<p align="center"><b>Next-Generation Intrusion Detection for the Banking Sector</b></p>

<p align="center"> <img src="https://img.shields.io/badge/Model-Random_Forest_Ensemble-616ae5?style=for-the-badge&logo=scikit-learn&logoColor=white" /> <img src="https://img.shields.io/badge/Domain-FinTech_Security-FFD700?style=for-the-badge&logo=bank&logoColor=black" /> <img src="https://img.shields.io/badge/Accuracy-99.2%25-2ea44f?style=for-the-badge" /> </p>

💎 Project Essence
In the high-stakes environment of banking, traditional rule-based firewalls often fail against sophisticated anomalies. NetGuard-RF implements a cognitive layer that learns from transaction patterns, distinguishing between legitimate banking operations and suspicious network intrusions with near-perfect precision.

🚀 Performance Dashboard
The model achieves a 100% Precision rate, crucial for banking to ensure no "False Positives" lock out legitimate customers.

<table border="0"> <tr style="background-color: transparent;"> <td width="50%"> <img src="https://raw.githubusercontent.com/HRISHIK123958/research/main/outputs/1_confusion_matrix.png" style="border-radius:10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);" /> <p align="center"><b>Classification Accuracy</b></p> </td> <td width="50%"> <img src="https://raw.githubusercontent.com/HRISHIK123958/research/main/outputs/3_roc_curve.png" style="border-radius:10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);" /> <p align="center"><b>Reliability Curve (AUC: 0.999)</b></p> </td> </tr> </table>

🧠 Explainable AI (XAI) & Security Logic
We utilize Feature Importance to bridge the gap between black-box ML and human-auditable security.

Financial Context: AmountNormalized is the highest predictor, identifying transaction anomalies.

Temporal Context: Hour analysis detects off-peak suspicious attempts.

<p align="center"> <img src="https://raw.githubusercontent.com/HRISHIK123958/research/main/outputs/2_feature_importance.png" width="90%" style="border-radius:15px;" /> </p>

📊 Operational Stability
Our testing confirms high stability even in imbalanced datasets typical of banking fraud detection.

<table border="0"> <tr> <td width="40%"> <img src="https://raw.githubusercontent.com/HRISHIK123958/research/main/outputs/4_prediction_distribution.png" width="100%" /> </td> <td width="60%"> <img src="https://raw.githubusercontent.com/HRISHIK123958/research/main/outputs/5_metrics_comparison.png" width="100%" /> </td> </tr> <tr> <td align="center"><b>Class Distribution</b></td> <td align="center"><b>Validation Scorecard</b></td> </tr> </table>

🔧 Deployment Schema
The system is optimized for Ultra-Low Latency inference.

Python

import joblib

# Initialize Cognitive Engine
engine = joblib.load('random_forest_firewall_model.joblib')

# Process Real-time Traffic
# Feature vector: [Amount, IsHighAmount, Hour, IsNight, Foreign, etc.]
decision = engine.predict(network_payload)

if decision == 'Suspicious':
    trigger_security_protocol()

<p align="center"><b>ADAMAS UNIVERSITY</b></p>

<p align="center"> <b> <a href="https://github.com/iArnabMaity">Arnab Maity</a> • <a href="https://github.com/HRISHIK123958">Hrishik Dey</a> • <a href="https://github.com/zainab9hasan">Zainab Hasan</a> • <a href="https://github.com/sohel762">Sohel Ali Mondal</a> • <a href="https://github.com/Sohelshaikh2301">Sk Sohel Rahaman</a> </b> </p>
<p align="center"> <img src="https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg" /> <img src="https://img.shields.io/badge/Contributions-Welcome-blue.svg" /> </p>
