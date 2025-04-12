class Counter:

    _counts = {}

    def __new__(cls, *args, **kwargs):
        raise RuntimeError("This is a utility class and cannot be instantiated.")

    @classmethod
    def get_count(cls, name: str = 'default') -> int:
        if name in cls._counts:
            cls._counts[name] += 1
        else:
            cls._counts[name] = 0
        return cls._counts[name]

    @classmethod
    def reset_count(cls, name: str = 'default') -> None:
        if name in cls._counts:
            del cls._counts[name]

    @classmethod
    def reset_all_counts(cls) -> None:
        cls._counts.clear()

    @classmethod
    def get_all_counts(cls) -> dict:
        """获取所有计数器副本"""
        return cls._counts.copy()