import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

class Logger() :
    path : str

    def __init__(self, log_path):

        log_dir = os.environ['LOG_DIR']
        
        if not os.path.exists(log_dir) :
            print(f'[CREATE LOG DIR] Log dir not set. Setting up at {log_dir}')
            os.mkdir(log_dir)

        if type(log_path) != str :
            raise TypeError("[ERROR LOGGER] log path should be str")
        
        self.path = os.path.join(log_dir, log_path)
        
        if os.path.exists(self.path) and os.path.isfile(self.path) :
            print(f"[USE LOG FILE] Logger using {log_path}")
        else :
            with open(self.path, 'w') as f :
                f.write("")
            print(f"[CREATE LOG FILE] Logger creating and using {self.path}")
    
    def log(self, text : str) :
        with open(self.path, 'a') as f :
            f.write(f"{datetime.today().strftime('%Y-%m-%d, %H:%M:%S:%f')} - {text}\n")