import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger

def test_logger():
    logger = get_logger("TestLogger")
    test_message = "This is a test log message."
    logger.info(test_message)

    log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "app.log")
    
    if not os.path.exists(log_file):
        print("FAILED: Log file was not created.")
        return

    with open(log_file, "r") as f:
        content = f.read()
        if test_message in content:
            print("SUCCESS: Log message found in file.")
        else:
            print(f"FAILED: Log message not found. Content:\n{content}")

if __name__ == "__main__":
    test_logger()
