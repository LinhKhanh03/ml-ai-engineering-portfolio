from app.ingestion.ingest import ingest
from app.core.chat import chat
from agents.core.chat import chat_agent
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":
    print("1. Ingest PDF")
    print("2. Chat with PDF")
    print("3. Chat with Agent")
    choice = input("\nChoose: ")
    if choice == "1":
        ingest()
    elif choice == "2":
        chat()
    elif choice == "3":
        chat_agent()
    else:
        print("Invalid choice")