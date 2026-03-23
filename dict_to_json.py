dct = eval(input("Enter a dictionary: "))

import json

json_str = json.dumps(dct, indent=4)
print("JSON representation:\n", json_str)
