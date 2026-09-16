import sys
import time

def print_slow(text, delay=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def main():
    print_slow("\033[96mWelcome to AIOS Terminal! Type 'help' for available commands.\033[0m")
    
    while True:
        try:
            choice = input("🚀 km1558 \033[91m->\033[0m \033[94mAIOS-LSFS\033[0m \033[91m>>\033[0m \033[92mDo you want to mount AIOS Semantic File System to a specific directory you want? By default, it will be mounted 'at your mount path'. [y/n]\033[0m ")
            if choice.lower() == 'y':
                path = input("Enter mount path: ")
                print_slow(f"Mounting AIOS-LSFS (Large Semantic File System) at {path}...")
                time.sleep(1)
                print_slow("Success: Ring-0 Storage Broker connected to Semantic VFS.")
                break
            elif choice.lower() == 'n':
                print_slow("Mounting AIOS-LSFS at default mount path /mnt/aios_lsfs...")
                time.sleep(1)
                print_slow("Success: Ring-0 Storage Broker connected to Semantic VFS.")
                break
        except KeyboardInterrupt:
            print()
            sys.exit(0)

if __name__ == "__main__":
    main()
