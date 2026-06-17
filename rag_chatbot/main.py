from app.ingestion.ingest import ingest
from app.chat import chat
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    print("1. Ingest PDF")
    print("2. Chat with PDF")
    choice = input("\nChoose: ")
    if choice == "1":
        ingest()
    elif choice == "2":
        chat()
    else:
        print("Invalid choice")