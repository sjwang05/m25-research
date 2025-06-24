import csv
import argparse

def create_food_csv(local_file, en_file, output_file):
    """
    Reads foreign language and English food names from separate files and combines them into a CSV.
    
    Args:
        local_file: Path to the foreign language food names file (required)
        en_file: Path to the English food names file (required)
        output_file: Path to the output CSV file (required)
    """
    try:
        # Read foreign language food names
        with open(local_file, 'r', encoding='utf-8') as f:
            local_foods = [line.strip() for line in f.readlines() if line.strip()]
        
        # Read English food names
        with open(en_file, 'r', encoding='utf-8') as f:
            en_foods = [line.strip() for line in f.readlines() if line.strip()]
        
        # Check if both files have the same number of items
        if len(local_foods) != len(en_foods):
            print(f"Warning: Number of items don't match! Foreign language: {len(local_foods)}, English: {len(en_foods)}")
            print("The CSV will contain as many rows as the shorter list.")
        
        # Create CSV file
        with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['local_food', 'en_food'])
            
            # Write food pairs
            for local, en in zip(local_foods, en_foods):
                writer.writerow([local, en])
        
        print(f"Successfully created {output_file}")
        print(f"Total rows written: {min(len(local_foods), len(en_foods))}")
        
        # Show first few rows as preview
        print("\nFirst 5 rows of the CSV:")
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < 6:  # Header + 5 data rows
                    print(f"{row[0]:<30} | {row[1]}")
                else:
                    break
                    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description='Combine foreign language and English food names from separate files into a CSV'
    )
    
    # Add positional arguments
    parser.add_argument('local_file', 
                        help='Path to the foreign language food names file')
    parser.add_argument('en_file', 
                        help='Path to the English food names file')
    parser.add_argument('output_file', 
                        help='Path to the output CSV file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the function with provided arguments
    create_food_csv(args.local_file, args.en_file, args.output_file)