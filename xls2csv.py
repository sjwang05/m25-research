import pandas as pd
import os

# List all Excel files in the current directory
excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx') or f.endswith('.xls')]

for file in excel_files:
    try:
        xl = pd.ExcelFile(file)
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, header=None)
            outname = f"{os.path.splitext(file)[0]}_{sheet}.csv"
            df.to_csv(outname, index=False, header=False)
            print(f"Saved: {outname}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
