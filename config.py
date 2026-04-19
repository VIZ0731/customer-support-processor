
import os
from dotenv import load_dotenv

load_dotenv() 

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b:free"

CATEGORY_TO_BRANCH = {
    "TECHNICAL": "technical",
    "BILLING": "billing",
    "GENERAL": "general",
    "COMPLAINT": "complaint"
}