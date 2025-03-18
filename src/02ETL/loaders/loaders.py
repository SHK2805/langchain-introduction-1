from src.Config.set_config import Config

config = Config()
if config.set():
    print("Environment variables set")
else:
    print("Environment variables NOT set")

