import torch
import time
import functools

def suppress_OOM(timeout: int = 3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f'OOM occurred, resume in {timeout} seconds')
                time.sleep(timeout)

        return wrapper
    return decorator

def clear_cache_on_leave(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        torch.cuda.empty_cache()
        return ret
        
    return wrapper