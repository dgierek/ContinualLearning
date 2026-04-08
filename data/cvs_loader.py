import os, re, pandas as pd

def load_cv_dataset(root_dir):
  rows = []

  for substance in os.listdir(root_dir):
    sub_path = os.path.join(root_dir, substance)
    if not os.path.isdir(sub_path):
      continue

    for scan_folder in os.listdir(sub_path):
      scan_path = os.path.join(sub_path, scan_folder)
      if not os.path.isdir(scan_path):
        continue

      # Extract scan rate from folder name
      scan_rate_match = re.search(r"scan_rate_(.*)_V_s", scan_folder)
      scan_rate = float(scan_rate_match.group(1).replace("_", "."))

      for file in os.listdir(scan_path):
        if not file.endswith(".xlsx"):
          continue

        file_path = os.path.join(scan_path, file)

        # Extract flow rate from filename
        flow_match = re.search(r"flow_rate_(.*)_ml_h", file)
        flow_rate = float(flow_match.group(1).replace("_", "."))

        # Read Excel
        df = pd.read_excel(file_path, skiprows=5)

        # Extract potential/current columns
        potential = df["Potential [V]:"].values
        current = df["Current [µA]:"].values

        rows.append({
          "substance": substance,
          "scan_rate": scan_rate,
          "flow_rate": flow_rate,
          "potential": potential.tolist(),
          "current"  : current.tolist()
        })

  df_rows = pd.DataFrame(rows)

  return df_rows