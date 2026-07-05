from datetime import datetime

#TODO: should we remove this method and file?
def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)