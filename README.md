<p align="center">🛡️ Implementation of AI-based Cognitive Firewall in Banking Sector</p>
<p align="center"> <img src="https://img.shields.io/badge/Model-Random_Forest-616ae5?style=for-the-badge&logo=scikit-learn&logoColor=white" /> <img src="https://img.shields.io/badge/Accuracy-99.2%25-success?style=for-the-badge" /> <img src="https://img.shields.io/badge/Precision-100%25-blue?style=for-the-badge" /> </p>

⚡ Project Vision
NetGuard-RF is a high-performance classification engine designed to distinguish between Normal and Suspicious network traffic. By leveraging ensemble learning, it provides a robust defense mechanism for automated security systems.

📊 Performance Analytics
The model was validated using a comprehensive suite of metrics to ensure zero-trust reliability.

<table border="0"> <tr> <td width="50%"> <p align="center"><b>Confusion Matrix</b></p> <img src="https://raw.githubusercontent.com/HRISHIK123958/research/main/outputs/1_confusion_matrix.png" alt="Confusion Matrix" width="100%" /> </td> <td width="50%"> <p align="center"><b>ROC / AUC Curve</b></p> <img src="https://raw.githubusercontent.com/HRISHIK123958/research/main/outputs/3_roc_curve.png" alt="ROC Curve" width="100%" /> </td> </tr> </table>

🧠 Explainable AI (XAI)
In security, the "Why" matters as much as the "What." Our feature analysis shows that Normalized Transaction Amounts and Time of Day are critical indicators of suspicious activity.

<p align="center"> <img src="https://raw.githubusercontent.com/HRISHIK123958/research/main/outputs/2_feature_importance.png" alt="Feature Importance" width="90%" /> </p>

📈 Data Distribution & Stability
We ensured the model remains stable across imbalanced datasets by monitoring prediction spreads and metric consistency.

<table border="0"> <tr> <td> <img src="https://raw.githubusercontent.com/HRISHIK123958/research/main/outputs/4_prediction_distribution.png" alt="Distribution" width="350" /> <p align="center"><b>Prediction Spread</b></p> </td> <td> <img src="https://raw.githubusercontent.com/HRISHIK123958/research/main/outputs/5_metrics_comparison.png" alt="Metrics" width="550" /> <p align="center"><b>Final Scorecard</b></p> </td> </tr> </table>

🚀 Integration
The model is production-ready via joblib.

Python

import joblib

# Load the brain
clf = joblib.load('random_forest_firewall_model.joblib')

# Classify traffic
# Returns: ['Normal'] or ['Suspicious']
result = clf.predict(live_traffic_data)
<p align="center"> <b>Developed by <a href="https://github.com/iArnabMaity">ARNAB MAITY</a>,  <a href="https://github.com/HRISHIK123958">HRISHIK DEY</a>,  <a hreF="https://github.com/zainab9hasan"> ZAINAB HASAN</a>,  <a href="https://github.com/Sohelshaikh2301">SK SOHEL RAHAMAN</a></b> | 2025</p>
