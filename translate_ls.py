# To run this code you need to install the following dependencies:
# uv pip install openai

import os
import sys
from openai import OpenAI

# Configurable source language
SRC_LANG = "Japanese"

# Configurable API settings
API_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ.get("API_KEY", os.environ.get("OPENROUTER_API_KEY"))
MODEL = "google/gemini-2.5-pro"


def process_chunk(client, model, chunk_lines):
    """Process a chunk of lines and return the translation."""
    # Join the lines into a single string
    foods = "\n".join(chunk_lines)
    
    # Create the prompt with the configured language
    prompt = (
        f"You are an expert translator who specializes in nutrition and food items. "
        f"Please translate the following list of foods from {SRC_LANG} to English. "
        f"Only provide direct romanizations/transliterations when an average English speaker "
        f"would be familiar with the romanization; otherwise, provide an English localization. "
        f"Provide only the translated list.\n\n"
        f"{foods}"
    )
    
    # Generate the response
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7, # slightly lower temp for (hopefully) more sane translations
    )
    
    return completion.choices[0].message.content.strip()


def main():
    if len(sys.argv) != 3 or not API_KEY:
        print("Usage: python script.py <input_file> <output_file>")
        print("\nEnvironment variables:")
        print("  API_KEY or OPENROUTER_API_KEY: Your API key (required)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
    
    print(f"Using API: {API_BASE_URL}")
    print(f"Using model: {MODEL}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            chunk = []
            line_count = 0
            total_chunks = 0
            
            for line in f_in:
                line = line.strip()
                if line:  # Skip empty lines
                    chunk.append(line)
                    line_count += 1
                
                # Process when we have 5 lines
                if len(chunk) == 5:
                    total_chunks += 1
                    print(f"Processing chunk {total_chunks} (lines {line_count-4} to {line_count})...")
                    try:
                        translation = process_chunk(client, MODEL, chunk)
                        f_out.write(translation)
                        if not translation.endswith('\n'):
                            f_out.write("\n")
                        f_out.flush()  # Ensure output is written immediately
                    except Exception as e:
                        print(f"Error processing chunk {total_chunks}: {e}")
                        # Write original lines as fallback
                        f_out.write("# Translation failed:\n")
                        f_out.write("\n".join(chunk))
                        f_out.write("\n")
                    chunk = []
            
            # Process any remaining lines
            if chunk:
                total_chunks += 1
                print(f"Processing final chunk {total_chunks} ({len(chunk)} lines)...")
                try:
                    translation = process_chunk(client, MODEL, chunk)
                    f_out.write(translation)
                    if not translation.endswith('\n'):
                        f_out.write("\n")
                except Exception as e:
                    print(f"Error processing final chunk: {e}")
                    # Write original lines as fallback
                    f_out.write("# Translation failed:\n")
                    f_out.write("\n".join(chunk))
        
        print(f"Translation complete! Processed {total_chunks} chunks.")
        print(f"Output saved to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
