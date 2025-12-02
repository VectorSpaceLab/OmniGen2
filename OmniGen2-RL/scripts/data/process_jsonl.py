import json
import os
import argparse

def convert_jsonl_paths(input_file, output_file, base_path, max_records=100000):
    """
    Convert JSONL file image paths from relative to absolute paths
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        base_path: Base directory path for converting relative paths
        max_records: Maximum number of records to process, default 10000
    """
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        count = 0
        for line in infile:
            if count >= max_records:
                break
                
            # Parse JSON line
            try:
                data = json.loads(line.strip())
                
                # Check if input_images field exists
                if "input_images" in data and isinstance(data["input_images"], list):
                    # Convert paths
                    new_paths = []
                    for path in data["input_images"]:
                        if isinstance(path, str) and path.startswith("images/"):
                            # Convert relative path to absolute path
                            new_path = path.replace("images/", f"{base_path}/images/")
                            new_paths.append(new_path)
                        else:
                            # Keep original path unchanged
                            new_paths.append(path)
                    
                    data["input_images"] = new_paths
                
                # Write converted data
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                count += 1
                
                # Print progress every 1000 records
                if count % 1000 == 0:
                    print(f"Processed {count} records")
                    
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
                continue
    
    print(f"Conversion completed! Total processed records: {count}")
    print(f"Output file: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL file image paths from relative to absolute")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file path")
    parser.add_argument("--base-path", "-b", required=True, help="Base directory path for converting relative paths")
    parser.add_argument("--max-records", "-m", type=int, default=100000, help="Maximum number of records to process (default: 100000)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Validate base path exists
    if not os.path.exists(args.base_path):
        print(f"Warning: Base path '{args.base_path}' does not exist")
    
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Base path: {args.base_path}")
    print(f"Max records: {args.max_records}")
    print("-" * 50)
    
    convert_jsonl_paths(args.input, args.output, args.base_path, args.max_records)

if __name__ == "__main__":
    main()