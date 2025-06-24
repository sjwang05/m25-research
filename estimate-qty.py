#!/usr/bin/env python3
"""
Script to estimate food quantities using OpenRouter API with parallel processing.
Processes food items from a CSV file and adds quantity estimates using batched async requests.
"""

import asyncio
import aiohttp
import json
import csv
import os
import logging
import argparse
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import time

# Configuration
API_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ.get("API_KEY", os.environ.get("OPENROUTER_API_KEY"))
MODEL = "google/gemini-2.5-flash"

# Language settings
SRC_LANG = "Japanese"
DEST_LANG = "English"

# API parameters
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_TOKENS = 4096

# Rate limiting configuration
MAX_CONCURRENT_REQUESTS = 10  # Adjust based on your API limits
RETRY_ATTEMPTS = 5
RETRY_DELAY = 1  # seconds

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QuantityRequest:
    """Represents a single quantity estimation request"""
    row_index: int
    japanese_foods: List[str]
    english_foods: List[str]
    original_row: Dict[str, Any]
    meal_time: str


@dataclass
class QuantityResult:
    """Represents a quantity estimation result"""
    row_index: int
    japanese_quantities: List[str]
    english_quantities: List[str]
    original_row: Dict[str, Any]
    success: bool
    error: Optional[str] = None


def create_prompt(japanese_foods: List[str], english_foods: List[str]) -> str:
    """Create the quantity estimation prompt"""
    prompt = (
        f"You are a nutrition expert who specializes in {SRC_LANG} cuisine and meal portions. "
        f"Given the following lists of food items from a single meal, estimate appropriate serving quantities.\n\n"
        f"{SRC_LANG} foods: {json.dumps(japanese_foods, ensure_ascii=False)}\n"
        f"{DEST_LANG} foods: {json.dumps(english_foods, ensure_ascii=False)}\n\n"
        f"Provide quantities for each food item. Return ONLY two lists:\n"
        f"First line: {SRC_LANG} quantities (e.g., [一人前, 一杯, 一個, 一切れ, 一本])\n"
        f"Second line: {DEST_LANG} quantities (e.g., [one serving, one bowl, one piece, one slice, one bottle])\n"
        f"Each list should be comma-separated and match the order of the input foods.\n"
        f"Use everyday units such as \"one slice\", \"one bottle\", or \"one bowl\" rather than "
        f"exact numerical quantities, even if the original food item's name contains exact measurements."
    )
    return prompt


def parse_quantity_response(response_text: str, expected_jp_count: int, expected_en_count: int) -> Tuple[List[str], List[str]]:
    """Parse the API response to extract quantities"""
    lines = response_text.strip().split('\n')
    
    if len(lines) >= 2:
        # Extract the two lists from the response
        jp_line = lines[0].strip()
        en_line = lines[1].strip()
        
        # Parse the comma-separated values
        # Remove brackets if present and split by comma
        jp_line = jp_line.strip('[]')
        en_line = en_line.strip('[]')
        
        jp_quantities = [q.strip().strip('"\'') for q in jp_line.split(',')]
        en_quantities = [q.strip().strip('"\'') for q in en_line.split(',')]
        
        # Validate counts
        if len(jp_quantities) == expected_jp_count and len(en_quantities) == expected_en_count:
            return jp_quantities, en_quantities
    
    # Return defaults if parsing fails
    return ["一人前"] * expected_jp_count, ["one serving"] * expected_en_count


async def estimate_quantities(session: aiohttp.ClientSession, request: QuantityRequest) -> QuantityResult:
    """Send quantity estimation request to OpenRouter API"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    prompt = create_prompt(request.japanese_foods, request.english_foods)
    
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
                    response_text = data['choices'][0]['message']['content']
                    
                    # Parse the response
                    jp_quantities, en_quantities = parse_quantity_response(
                        response_text,
                        len(request.japanese_foods),
                        len(request.english_foods)
                    )
                    
                    logger.info(f"Successfully estimated quantities for row {request.row_index} ({request.meal_time})")
                    
                    return QuantityResult(
                        row_index=request.row_index,
                        japanese_quantities=jp_quantities,
                        english_quantities=en_quantities,
                        original_row=request.original_row,
                        success=True
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"API error for row {request.row_index}: {response.status} - {error_text}")
                    
                    if attempt < RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    
                    # Use defaults on final failure
                    return QuantityResult(
                        row_index=request.row_index,
                        japanese_quantities=["一人前"] * len(request.japanese_foods),
                        english_quantities=["one serving"] * len(request.english_foods),
                        original_row=request.original_row,
                        success=False,
                        error=f"API error: {response.status} - {error_text}"
                    )
                    
        except Exception as e:
            logger.error(f"Exception for row {request.row_index}: {str(e)}")
            
            if attempt < RETRY_ATTEMPTS - 1:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                continue
            
            # Use defaults on exception
            return QuantityResult(
                row_index=request.row_index,
                japanese_quantities=["一人前"] * len(request.japanese_foods),
                english_quantities=["one serving"] * len(request.english_foods),
                original_row=request.original_row,
                success=False,
                error=f"Exception: {str(e)}"
            )
    
    # Should never reach here, but just in case
    return QuantityResult(
        row_index=request.row_index,
        japanese_quantities=["一人前"] * len(request.japanese_foods),
        english_quantities=["one serving"] * len(request.english_foods),
        original_row=request.original_row,
        success=False,
        error="Unknown error after all retries"
    )


async def process_batch(session: aiohttp.ClientSession, requests: List[QuantityRequest]) -> List[QuantityResult]:
    """Process a batch of quantity estimation requests concurrently"""
    tasks = [estimate_quantities(session, req) for req in requests]
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
                try:
                    # Parse the JSON strings
                    jp_foods = json.loads(row.get('description_local', '[]'))
                    en_foods = json.loads(row.get('description', '[]'))
                    meal_time = row.get('meal_time', 'unknown')
                    
                    if isinstance(jp_foods, list) and isinstance(en_foods, list) and jp_foods and en_foods:
                        requests.append(QuantityRequest(
                            row_index=idx,
                            japanese_foods=jp_foods,
                            english_foods=en_foods,
                            original_row=row,
                            meal_time=meal_time
                        ))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON in row {idx}: {e}")
                    continue
        
        logger.info(f"Loaded {len(requests)} rows for quantity estimation from {input_file}")
        
    except FileNotFoundError:
        logger.error(f"File not found: {input_file}")
        return
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        return
    
    # Process requests in batches
    results = []
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        # Process in batches to respect rate limits
        for i in range(0, len(requests), MAX_CONCURRENT_REQUESTS):
            batch = requests[i:i + MAX_CONCURRENT_REQUESTS]
            batch_num = i // MAX_CONCURRENT_REQUESTS + 1
            total_batches = (len(requests) - 1) // MAX_CONCURRENT_REQUESTS + 1
            
            logger.info(f"Processing batch {batch_num} of {total_batches}")
            
            batch_results = await process_batch(session, batch)
            results.extend(batch_results)
            
            # Small delay between batches to avoid rate limiting
            if i + MAX_CONCURRENT_REQUESTS < len(requests):
                await asyncio.sleep(0.5)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing complete in {elapsed_time:.2f} seconds")
    
    # Save results as CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Get all fieldnames from original CSV
        if results:
            fieldnames = list(results[0].original_row.keys())
            
            # Ensure unit columns exist
            if 'unit_local' not in fieldnames:
                fieldnames.append('unit_local')
            if 'unit' not in fieldnames:
                fieldnames.append('unit')
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Sort results by original row index
            for result in sorted(results, key=lambda x: x.row_index):
                row_data = result.original_row.copy()
                
                # Add quantity data
                row_data['unit_local'] = json.dumps(result.japanese_quantities, ensure_ascii=False)
                row_data['unit'] = json.dumps(result.english_quantities, ensure_ascii=False)
                
                writer.writerow(row_data)
    
    # Log summary statistics
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Total rows processed: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if failed > 0:
        logger.warning("Some rows failed to process. Default quantities were used.")
    
    # Show sample of results
    print("\nSample of estimated quantities:")
    sample_size = min(3, len(results))
    for i in range(sample_size):
        result = results[i]
        print(f"\nMeal {i+1} (Row {result.row_index + 1}, {result.original_row.get('meal_time', 'unknown')}):")
        
        # Display up to 3 items from each meal
        display_count = min(3, len(result.japanese_quantities), len(result.english_quantities))
        for j in range(display_count):
            if j < len(result.japanese_quantities):
                jp_food = json.loads(result.original_row['description_local'])[j]
                print(f"  {jp_food} → {result.japanese_quantities[j]}")
            if j < len(result.english_quantities):
                en_food = json.loads(result.original_row['description'])[j]
                print(f"  {en_food} → {result.english_quantities[j]}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Estimate food quantities using OpenRouter API with parallel processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.csv output.csv
  
Environment:
  Set OPENROUTER_API_KEY or API_KEY environment variable with your API key.
  
Configuration:
  You can adjust MAX_CONCURRENT_REQUESTS in the script to control parallelism.
  Current setting: {} concurrent requests
        """.format(MAX_CONCURRENT_REQUESTS)
    )
    
    parser.add_argument('input_csv', 
                        help='Input CSV file with food descriptions')
    parser.add_argument('output_csv', 
                        help='Output CSV file with quantity estimates')
    
    args = parser.parse_args()
    
    # Run the async main function
    asyncio.run(main(args.input_csv, args.output_csv))
