import time

from llm_planner.logger import Logger

# Create a custom logger
logger = Logger('Timing')


def timing(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Determine if the function is a method of a class
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            logger.info(
                f"Execution of {class_name}.{func.__name__} took {end_time - start_time:.6f} seconds"
            )
        else:
            logger.info(
                f"Execution of {func.__name__} took {end_time - start_time:.6f} seconds"
            )

        return result

    return wrapper


import inspect


def test_timing(func):

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        # Retrieve the function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Format the arguments
        args_str = ""
        if 'tag' in bound_args.arguments:
            args_str = bound_args.arguments['tag']

        # Determine if the function is a method of a class
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            logger.info(
                f"{class_name}.{func.__name__} | {args_str}| Execution Time | {end_time - start_time:.6f} | seconds"
            )
        else:
            logger.info(
                f"{func.__name__} | {args_str}| Execution Time | {end_time - start_time:.6f} | seconds"
            )

        return result

    return wrapper


def is_typed_dict(obj: object) -> bool:
    return hasattr(obj, '__annotations__') and isinstance(
        obj.__class__, type) and issubclass(obj.__class__, dict)
