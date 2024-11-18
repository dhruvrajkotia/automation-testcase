import logging

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger; this can be reused."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Create a logger instance for the evaluate service
evaluate_service_logger = setup_logger('evaluate_service', 'evaluate_service.log')
