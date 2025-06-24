import asyncio
import aiohttp
import json
import csv
import os
import logging
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

# Configuration
API_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ.get("API_KEY", os.environ.get("OPENROUTER_API_KEY"))
MODEL = "google/gemini-2.5-flash"

# Translation parameters
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 4096

# Source language configuration
SRC_LANG = "Japanese"

# Translation notes
NOTES = """Note:
- In Japanese, the name of a dish is sometimes expressed in the way the dish is cooked, \
so the English name should reflect the name of the dish, not its cooking method.
"""

# Main translation prompt template
MAIN_PROMPT = (
    "You are an experienced translator of Japanese manga and LNs. "
    "Provide concise, specific, and accurate translations of the following list of foods from {src_lang} to English, where "
    "each food item is a single item in a meal. Provide the translated list with each item on its own line. "
    "Only provide direct romanizations/transliterations when an average English speaker would be familiar with the romanization; "
    "otherwise, provide an English localization. Provide only the translated list.\n\n"
    "{foods_formatted}\n\n"
    "{notes}"
)

# Rate limiting configuration
MAX_CONCURRENT_REQUESTS = 10  # Adjust based on your API limits
RETRY_ATTEMPTS = 5
RETRY_DELAY = 1  # seconds

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TranslationRequest:
    """Represents a single translation request"""
    row_index: int
    foods_japanese: List[str]
    original_row: Dict[str, Any]


@dataclass
class TranslationResult:
    """Represents a translation result"""
    row_index: int
    foods_japanese: List[str]
    foods_english: List[str]
    original_row: Dict[str, Any]
    success: bool
    error: Optional[str] = None


def create_prompt(foods_list: List[str]) -> str:
    """Create the translation prompt"""
    foods_formatted = "\n".join(foods_list)
    prompt = MAIN_PROMPT.format(
        src_lang=SRC_LANG,
        foods_formatted=foods_formatted,
        notes=NOTES
    )
    return prompt


async def translate_foods(session: aiohttp.ClientSession, request: TranslationRequest) -> TranslationResult:
    """Send translation request to OpenRouter API"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    prompt = create_prompt(request.foods_japanese)
    
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "reasoning": {
            "enabled": True
        }
    }
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with session.post(
                f"{API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    translated_text = data['choices'][0]['message']['content']
                    
                    # Parse the translated list (assuming one item per line)
                    translated_foods = [line.strip() for line in translated_text.strip().split('\n') if line.strip()]

                    # Validate length matches original
                    if len(translated_foods) != len(request.foods_japanese):
                        error_msg = (f"Length mismatch: original list has {len(request.foods_japanese)} items, "
                                     f"translated list has {len(translated_foods)} items.")
                        logger.error(f"Validation error for row {request.row_index}: {error_msg}")
                        if attempt < RETRY_ATTEMPTS - 1:
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                            continue
                        else:
                            return TranslationResult(
                                row_index=request.row_index,
                                foods_japanese=request.foods_japanese,
                                foods_english=translated_foods,
                                original_row=request.original_row,
                                success=False,
                                error=error_msg
                            )
                    
                    logger.info(f"Successfully translated row {request.row_index}")
                    
                    return TranslationResult(
                        row_index=request.row_index,
                        foods_japanese=request.foods_japanese,
                        foods_english=translated_foods,
                        original_row=request.original_row,
                        success=True
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"API error for row {request.row_index}: {response.status} - {error_text}")
                    
                    if attempt < RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    
                    return TranslationResult(
                        row_index=request.row_index,
                        foods_japanese=request.foods_japanese,
                        foods_english=[],
                        original_row=request.original_row,
                        success=False,
                        error=f"API error: {response.status} - {error_text}"
                    )
                    
        except Exception as e:
            logger.error(f"Exception for row {request.row_index}: {str(e)}")
            
            if attempt < RETRY_ATTEMPTS - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
                
            return TranslationResult(
                row_index=request.row_index,
                foods_japanese=request.foods_japanese,
                foods_english=[],
                original_row=request.original_row,
                success=False,
                error=f"Exception: {str(e)}"
            )
    
    # If all attempts fail and no return has occurred, return a generic failure result
    # Should never happen with RETRY_ATTEMPTS > 0
    return TranslationResult(
        row_index=request.row_index,
        foods_japanese=request.foods_japanese,
        foods_english=[],
        original_row=request.original_row,
        success=False,
        error="Unknown error: translation failed after all retry attempts."
    )


async def process_batch(session: aiohttp.ClientSession, requests: List[TranslationRequest]) -> List[TranslationResult]:
    """Process a batch of translation requests concurrently"""
    tasks = [translate_foods(session, req) for req in requests]
    return await asyncio.gather(*tasks)


async def main(input_file: str, output_file: str):
    """Main function to process the CSV file"""
    
    if not API_KEY:
        logger.error("API_KEY or OPENROUTER_API_KEY environment variable not set")
        return
    
    # Read the CSV file
    requests = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for idx, row in enumerate(reader):
                # Parse the description_local column
                description_local = row.get('description_local', '')
                
                try:
                    # The description_local appears to be a JSON-encoded list
                    foods_japanese = json.loads(description_local)
                    
                    if isinstance(foods_japanese, list) and foods_japanese:
                        requests.append(TranslationRequest(
                            row_index=idx,
                            foods_japanese=foods_japanese,
                            original_row=row
                        ))
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON in row {idx}: {description_local}")
                    continue
        
        logger.info(f"Loaded {len(requests)} rows for translation from {input_file}")
        
    except FileNotFoundError:
        logger.error(f"File not found: {input_file}")
        return
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return
    
    # Process requests in batches
    results = []
    
    async with aiohttp.ClientSession() as session:
        # Process in batches to respect rate limits
        for i in range(0, len(requests), MAX_CONCURRENT_REQUESTS):
            batch = requests[i:i + MAX_CONCURRENT_REQUESTS]
            logger.info(f"Processing batch {i // MAX_CONCURRENT_REQUESTS + 1} of {(len(requests) - 1) // MAX_CONCURRENT_REQUESTS + 1}")
            
            batch_results = await process_batch(session, batch)
            results.extend(batch_results)
            
            # Small delay between batches to avoid rate limiting
            if i + MAX_CONCURRENT_REQUESTS < len(requests):
                await asyncio.sleep(0.5)
    
    # Save results as CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Get all fieldnames from original CSV plus the translation
        if results:
            fieldnames = list(results[0].original_row.keys())
            if 'description' not in fieldnames:
                fieldnames.append('description')
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in sorted(results, key=lambda x: x.row_index):
                row_data = result.original_row.copy()
                if result.success:
                    # Write the English translations to the 'description' column as a JSON list
                    row_data['description'] = json.dumps(result.foods_english, ensure_ascii=False)
                else:
                    row_data['description'] = f"ERROR: {result.error}"
                writer.writerow(row_data)
    
    logger.info(f"Translation complete. Translations saved to {output_file} (CSV format)")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Translate NutriBench Japanese food names to English using OpenRouter API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.csv output.csv
  
Environment:
  Set OPENROUTER_API_KEY or API_KEY environment variable with your API key.
        """
    )
    
    parser.add_argument('input_csv', 
                        help='Input CSV file with Japanese food descriptions')
    parser.add_argument('output_file', 
                        help='Output file (CSV only)')
    
    args = parser.parse_args()
    
    # Run the async main function
    asyncio.run(main(args.input_csv, args.output_file))
