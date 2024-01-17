import os
from dotenv import load_dotenv
import memory_profiler

load_dotenv()


def is_profiling_enabled():
    enable_profiling = os.getenv('ENABLE_PROFILING', "False")
    return True if enable_profiling == "True" else False

def conditional_profile(func):
    if is_profiling_enabled():
        return memory_profiler.profile(func)
    else:
        return func
