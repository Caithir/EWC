from config import run_settings
from experiments import standard, fisher

def main():
    """Entry point for bash scripts
        No changes should be made here
        all customization should occur in the config
    """
    experiments = {
        "standard": standard,
        "fisher": fisher,
    }
    for exp in run_settings['experiments']:
        experiments[exp]()



if __name__ == "__main__":
    main()
