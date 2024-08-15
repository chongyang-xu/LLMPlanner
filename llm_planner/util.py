from enum import Enum
import sys, time
import inspect
import pynvml

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


def gpu_version_to_name(major, minor):
    if major == 8:
        return 'Ampere'
    elif major == 9:
        return 'Hopper'
    elif major == 7:
        return 'Volta' if minor == 0 else 'Turing'
    elif major == 6:
        return 'Pascal'
    elif major == 5:
        return 'Maxwell'
    elif major == 3:
        return 'Kepler'
    elif major == 2:
        return 'Fermi'
    else:
        return 'Unknown'


def get_gpu_name():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count < 1:
        return 'Unknown'
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
    arch_name = gpu_version_to_name(major, minor)

    pynvml.nvmlShutdown()

    return arch_name


def report_system_config():
    print('-' * 50)
    print('Python Version:', sys.version)
    print('-' * 50)


class ModelType(Enum):
    LOCAL = "Local"
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"


def model_type(model: str):
    if model.startswith("gpt-"):
        return ModelType.OPENAI
    elif model.startswith("claude-"):
        return ModelType.ANTHROPIC
    elif model.startswith("/"):
        return ModelType.LOCAL
    else:
        return False
