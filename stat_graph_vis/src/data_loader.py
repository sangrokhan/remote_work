import pandas as pd
import numpy as np
import os

def generate_sample_data(output_path):
    np.random.seed(42)
    n_samples = 1000
    
    # Base physical parameters
    tx_power = np.random.normal(30, 5, n_samples)
    antenna_tilt = np.random.normal(6, 2, n_samples)
    
    # 수신 품질 (물리 파라미터에 의존)
    rsrp = tx_power - 100 - (antenna_tilt * 0.5) + np.random.normal(0, 2, n_samples)
    sinr = rsrp + 20 + np.random.normal(0, 5, n_samples)
    cqi = np.clip((sinr + 10) / 3, 1, 15).astype(int)
    
    # 성능 지표 (수신 품질에 의존)
    throughput = (cqi * 50) + np.random.normal(0, 100, n_samples)
    latency = 100 - (sinr * 2) + np.random.normal(0, 10, n_samples)
    packet_loss = np.clip(10 - (sinr / 2), 0, 100)
    
    df = pd.DataFrame({
        'TX_Power': tx_power,
        'Antenna_Tilt': antenna_tilt,
        'RSRP': rsrp,
        'SINR': sinr,
        'CQI': cqi,
        'Throughput': throughput,
        'Latency': latency,
        'Packet_Loss_Rate': packet_loss
    })
    
    df.to_csv(output_path, index=False)
    print(f"Sample data generated at {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "kpi_samples.csv")
    generate_sample_data(data_path)
