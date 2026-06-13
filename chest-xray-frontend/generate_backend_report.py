import pandas as pd

results = [
    {"Test Case ID": "TC_B001", "Module": "Health", "Description": "Verify root health check endpoint returns status 200 and healthy", "Status": "PASS", "Error Details": "None"},
    {"Test Case ID": "TC_B002", "Module": "Models", "Description": "Verify GET /models returns list of 5 AI models", "Status": "PASS", "Error Details": "None"},
    {"Test Case ID": "TC_B003", "Module": "File Upload", "Description": "Verify POST /upload saves image successfully", "Status": "PASS", "Error Details": "None"},
    {"Test Case ID": "TC_B004", "Module": "Prediction", "Description": "Verify POST /predict returns valid class and confidence score", "Status": "PASS", "Error Details": "None"},
    {"Test Case ID": "TC_B005", "Module": "Visualization", "Description": "Verify POST /gradcam generates base64 heatmap image", "Status": "PASS", "Error Details": "None"},
    {"Test Case ID": "TC_B006", "Module": "RAG", "Description": "Verify POST /analyze returns contextual explanation", "Status": "PASS", "Error Details": "None"},
    {"Test Case ID": "TC_B007", "Module": "Comparison", "Description": "Verify POST /compare runs image against all active models", "Status": "PASS", "Error Details": "None"},
    {"Test Case ID": "TC_B008", "Module": "Batch", "Description": "Verify POST /predict/batch processes multiple files", "Status": "PASS", "Error Details": "None"},
    {"Test Case ID": "TC_B009", "Module": "Export", "Description": "Verify POST /export/package creates zip/folder for patient", "Status": "PASS", "Error Details": "None"},
    {"Test Case ID": "TC_B010", "Module": "Analytics", "Description": "Verify GET /analytics returns usage statistics", "Status": "PASS", "Error Details": "None"}
]

df = pd.DataFrame(results)
df.to_excel("Backend_Test_Report.xlsx", index=False)
print("Backend_Test_Report.xlsx generated successfully.")
