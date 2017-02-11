
def cache_by_first_parameter(function_to_cache):
    cache = {}

    def wrapper(first, *args, **kwargs):
        if first in cache:
            return cache[first]
        result = function_to_cache(first, *args, **kwargs)
        cache[first] = result
        return result
    return wrapper
