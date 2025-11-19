def unwrap_session(session_or_manager):
    """
    FastAPI injects a _GeneratorContextManager when using @contextmanager.
    This unwraps it into an actual SQLAlchemy Session.
    """
    
    if hasattr(session_or_manager, "__enter__"):
        return session_or_manager.__enter__()
    return session_or_manager
