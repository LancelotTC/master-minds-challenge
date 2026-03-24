import ast
import json


raw_value = input("Enter a Python dict or JSON object: ").strip()

try:
    parsed_value = json.loads(raw_value)
except json.JSONDecodeError:
    parsed_value = ast.literal_eval(raw_value)

json_str = json.dumps(parsed_value, indent=4, ensure_ascii=False)
print("JSON representation:\n", json_str)
