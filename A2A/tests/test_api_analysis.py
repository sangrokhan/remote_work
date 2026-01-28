from fastapi.testclient import TestClient
import sys
import os
import shutil

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from scripts.generate_mock_data import generate_data

client = TestClient(app)

def test_analyze_parquet_endpoint():
    # 1. Generate Mock Data
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    parquet_file = os.path.join(output_dir, "test_analysis.parquet")
    
    print(f"Generating mock data at {parquet_file}...")
    generate_data(num_days=1, anomaly_ratio=0.1, output_path=parquet_file)
    
    # Ensure file exists
    assert os.path.exists(parquet_file)
    
    # 2. Call API
    print("Calling API...")
    response = client.post(
        "/api/analyze-parquet",
        json={"files": [parquet_file]}
    )
    
    # 3. Verify Response
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    analysis_text = data["analysis"]
    
    print("Analysis Result:")
    print(analysis_text)
    
    # Check for expected keywords in analysis
    assert "Analysis of" in analysis_text
    assert "Shape:" in analysis_text
    assert "Columns:" in analysis_text
    assert "Basic Statistics" in analysis_text
    
    # Cleanup (Optional, maybe keep for inspection)
    # os.remove(parquet_file)

if __name__ == "__main__":
    try:
        test_analyze_parquet_endpoint()
        print("\n✅ Test Passed Successfully!")
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
