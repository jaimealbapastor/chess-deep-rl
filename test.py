import logging

# Configure logging
logging.basicConfig(
    filename='log_output.txt',  # Output file
    filemode='w',              # Overwrite the file each run; use 'a' for append mode
    level=logging.DEBUG,       # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format of log messages
)

# Log some messages
logging.debug('This is a debug message.')
logging.info('This is an info message.')
logging.warning('This is a warning message.')
logging.error('This is an error message.')
logging.critical('This is a critical message.')

print("Logging complete. Check 'log_output.txt'")
