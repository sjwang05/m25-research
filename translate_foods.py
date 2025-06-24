import pandas as pd
import json
import argparse

def load_translation_dictionary(dict_file):
    """Load the foreign language-English food dictionary into a dict"""
    df = pd.read_csv(dict_file)
    # Create a dictionary for easy lookup
    translation_dict = dict(zip(df['local_food'], df['en_food']))
    return translation_dict

def translate_meal(meal_list, translation_dict):
    """Translate a list of foreign language food items to English"""
    translated_meal = []
    untranslated_items = []
    
    for item in meal_list:
        if item in translation_dict:
            translated_meal.append(translation_dict[item])
        else:
            # Item not found in dictionary
            translated_meal.append(f"[{item}]")  # Mark untranslated items
            untranslated_items.append(item)
    
    return translated_meal, untranslated_items

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate hospital meals from a foreign language to English')
    parser.add_argument('input_file', help='Input CSV file with meal descriptions in a foreign language')
    parser.add_argument('output_file', help='Output CSV file with English translations')
    parser.add_argument('-d', '--dict', help='Dictionary CSV file')
    
    args = parser.parse_args()
    
    # Load translation dictionary
    print(f"Loading translation dictionary from {args.dict}...")
    translation_dict = load_translation_dictionary(args.dict)
    print(f"Loaded {len(translation_dict)} translations")
    
    # Load meals data
    print(f"\nLoading meals data from {args.input_file}...")
    meals_df = pd.read_csv(args.input_file)
    print(f"Loaded {len(meals_df)} meals")
    
    # Translate meals
    print("\nTranslating meals...")
    translated_meals = []
    all_untranslated = set()
    
    for idx, row in meals_df.iterrows():
        # Parse the JSON array from description_local
        try:
            meal_items = json.loads(row['description_local'])
        except:
            print(f"Error parsing meal at row {idx}")
            translated_meals.append([])
            continue
        
        # Translate the meal
        translated_meal, untranslated = translate_meal(meal_items, translation_dict)
        translated_meals.append(translated_meal)
        all_untranslated.update(untranslated)
    
    # Add translated meals to dataframe
    meals_df['description'] = [json.dumps(meal, ensure_ascii=False) for meal in translated_meals]
    
    # Save results
    print(f"\nSaving translated meals to {args.output_file}...")
    meals_df.to_csv(args.output_file, index=False)
    
    # Print summary
    print(f"\nTranslation complete!")
    print(f"Total meals translated: {len(translated_meals)}")
    
    if all_untranslated:
        print(f"\nItems not found in dictionary ({len(all_untranslated)}):")
        for item in sorted(all_untranslated):
            print(f"  - {item}")
    
    # Show a few examples
    print("\n--- Example Translations ---")
    for i in range(min(3, len(meals_df))):
        print(f"\nMeal {i+1} ({meals_df.iloc[i]['meal_time']}):")
        original = json.loads(meals_df.iloc[i]['description_local'])
        translated = translated_meals[i]
        print("  Original:", original)
        print("  Translated:", translated)

if __name__ == "__main__":
    main()
