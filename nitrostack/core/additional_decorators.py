import time
import inspect
from functools import wraps
from typing import Callable, Any, Dict, List, Tuple
from nitrostack.core.context import ExecutionContext

def copy_mcp_attributes(src: Any, dst: Any) -> None:
    """Helper to copy all _mcp_ attributes from src function to dst function."""
    for attr in dir(src):
        if attr.startswith("_mcp_"):
            setattr(dst, attr, getattr(src, attr))

def cache(ttl: int = 60):
    """
    Caches method outputs for a specific TTL (in seconds).
    Skips ExecutionContext parameters when generating the cache key.
    """
    def decorator(func: Callable):
        cache_store: Dict[Tuple[Any, ...], Tuple[Any, float]] = {}
        setattr(func, "_mcp_cache_ttl", ttl)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Construct a cache key by filtering out ExecutionContext and self
            filtered_args = []
            for arg in args:
                # Skip self if it has class attributes or is the instance, skip ExecutionContext
                if isinstance(arg, ExecutionContext) or type(arg).__name__ == "ExecutionContext":
                    continue
                filtered_args.append(arg)

            filtered_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, ExecutionContext) or type(v).__name__ == "ExecutionContext":
                    continue
                filtered_kwargs[k] = v

            # Standardize key (args and kwargs)
            key = (tuple(filtered_args), frozenset(filtered_kwargs.items()))

            now = time.time()
            if key in cache_store:
                val, expiry = cache_store[key]
                if now < expiry:
                    return val

            result = await func(*args, **kwargs)
            cache_store[key] = (result, now + ttl)
            return result

        copy_mcp_attributes(func, wrapper)
        setattr(wrapper, "_mcp_cache_ttl", ttl)
        return wrapper
    return decorator

def rate_limit(max: int, window: int):
    """
    Rate limits method calls to max calls per window (in seconds).
    """
    def decorator(func: Callable):
        calls: List[float] = []

        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal calls
            now = time.time()
            # Filter timestamps in the window
            calls = [t for t in calls if now - t < window]
            if len(calls) >= max:
                raise ValueError(f"Rate limit exceeded. Maximum of {max} calls allowed every {window} seconds.")
            
            calls.append(now)
            return await func(*args, **kwargs)

        copy_mcp_attributes(func, wrapper)
        return wrapper
    return decorator


class HealthCheckRegistry:
    # Maps name -> (unbound_func, class_type)
    _checks: Dict[str, Tuple[Callable, Any]] = {}
    # Maps name -> bound_callable
    _bound_checks: Dict[str, Callable[[], bool]] = {}

    @classmethod
    def register(cls, name: str, func: Callable, class_type: Any = None) -> None:
        cls._checks[name] = (func, class_type)

    @classmethod
    def bind_instance(cls, name: str, func: Callable, instance: Any) -> None:
        # Bind the method to the resolved instance
        import inspect
        if inspect.ismethod(func):
            cls._bound_checks[name] = func
        else:
            cls._bound_checks[name] = lambda: func(instance)

    @classmethod
    def get_checks(cls) -> Dict[str, Callable[[], bool]]:
        return cls._bound_checks

    @classmethod
    async def run_all(cls) -> Dict[str, str]:
        """Run every registered health check and return {name: status}.

        Coroutine checks are awaited, sync checks are called directly, and a
        check that raises reports its own error without affecting the others.
        """
        results: Dict[str, str] = {}
        for name, check in list(cls._bound_checks.items()):
            try:
                outcome = check()
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                results[name] = "healthy" if outcome else "unhealthy"
            except Exception as exc:  # noqa: BLE001 - one bad check must not hide the rest
                results[name] = f"error: {exc}"
        return results

def health_check(name: str):
    """
    Decorator to mark a service or controller method as a health check.
    """
    def decorator(func: Callable):
        func._mcp_health_check_name = name
        # We don't have the class type here yet, but we will register it.
        # It will be bound during discovery in the McpApplicationFactory.
        HealthCheckRegistry.register(name, func)
        return func
    return decorator
