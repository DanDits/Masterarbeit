_internal_names = "__internal_names__"
_cache_pool = []


def clear_caches_by_name(name):
    for cache in _cache_pool:
        if name in cache[_internal_names]:
            names = cache[_internal_names]
            cache.clear()
            cache[_internal_names] = names


def cache_by_first_parameter(cache_names=None):
    def cacher(function_to_cache):
        cache = {_internal_names: ([] if cache_names is None else cache_names)}
        _cache_pool.append(cache)

        def wrapper(first, *args, **kwargs):
            if first in cache:
                return cache[first]
            result = function_to_cache(first, *args, **kwargs)
            cache[first] = result
            return result
        return wrapper
    return cacher
