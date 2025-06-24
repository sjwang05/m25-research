import csv
import io
import re
import ast
import sys

def extract_unique_foods(csv_data):
    """
    Parses CSV data to extract, clean, and deduplicate food items from
    the 'description_local' column.

    Args:
        csv_data (str): A string containing the CSV data.

    Returns:
        list: A sorted list of unique food item names.
    """
    # Use a set to store unique food items to automatically handle duplicates.
    unique_foods = set()

    # Use io.StringIO to treat the string data as a file for the csv reader
    csv_file = io.StringIO(csv_data)
    
    # Use DictReader to easily access columns by name
    reader = csv.DictReader(csv_file)

    for row in reader:
        # The 'description_local' column contains a string representation of a list.
        # We use ast.literal_eval() to safely parse this string into a Python list.
        try:
            food_list_str = row['description_local']
            # Safely evaluate the string to a list
            food_items = ast.literal_eval(food_list_str)

            if not isinstance(food_items, list):
                continue
                
        except (ValueError, SyntaxError, KeyError):
            # Skip rows where the column is missing or the format is incorrect
            print(f"Warning: Could not parse row: {row}")
            continue

        for item in food_items:
            if item:
                unique_foods.add(item)

    # Convert the set to a sorted list for a consistent output order.
    return sorted(list(unique_foods))

# --- Main execution block ---

if len(sys.argv) < 3:
    print("Usage: python ls_foods.py <input_csv_file> <output_file>")
    sys.exit(1)

# Define the input and output filenames
input_filename = sys.argv[1]
output_filename = sys.argv[2]

try:
    # Read the content from the specified CSV file.
    with open(input_filename, 'r', encoding='utf-8') as f:
        csv_content = f.read()
    
    # Process the file content to get the deduplicated list.
    deduplicated_foods = extract_unique_foods(csv_content)

    # Write the deduplicated list to the output file (one item per line)
    with open(output_filename, 'w', encoding='utf-8') as out_f:
        for food in deduplicated_foods:
            out_f.write(f"{food}\n")

    # Print a summary to the console
    print(f"Deduplicated list of food items from '{input_filename}' written to '{output_filename}'.")
    print(f"Total unique food items: {len(deduplicated_foods)}")
    if not deduplicated_foods:
        print("No food items were found or extracted.")

except FileNotFoundError:
    print(f"Error: The file '{input_filename}' was not found. Please make sure it's in the same directory as the script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
