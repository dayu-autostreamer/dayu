import threading
from typing import Any, Dict, Hashable, Optional, Type


class GlobalInstanceManager:
    """
    A thread-safe global instance manager that provides shared access to instances
    based on class type and instance identifiers. This enables:
    - Shared instances for same class + same ID across modules
    - Different instances for same class + different IDs
    - Automatic instance creation on first access
    - Explicit resource management

    The registry structure: {class_type: {instance_id: instance}}
    """
    _registry: Dict[Type, Dict[Optional[Hashable], Any]] = {}
    _lock = threading.RLock()

    @classmethod
    def get_instance(
            cls,
            instance_cls: Type,
            instance_id: Optional[Hashable] = None,
            *args,
            **kwargs
    ) -> Any:
        """
        Retrieve or create an instance of the specified class associated with the given ID.

        Args:
            instance_cls: Class type of the instance to manage
            instance_id: Unique identifier for instance grouping (None for default group)
            *args: Positional arguments for class constructor
            **kwargs: Keyword arguments for class constructor

        Returns:
            Existing or newly created instance of the specified class
        """
        with cls._lock:
            # Initialize class registry if not present
            if instance_cls not in cls._registry:
                cls._registry[instance_cls] = {}

            # Get class-specific registry
            class_registry = cls._registry[instance_cls]

            # Create instance if not exists
            if instance_id not in class_registry:
                class_registry[instance_id] = instance_cls(*args, **kwargs)

            return class_registry[instance_id]

    @classmethod
    def release_instance(
            cls,
            instance_cls: Type,
            instance_id: Optional[Hashable] = None
    ) -> None:
        """
        Release a specific instance from the registry.

        Args:
            instance_cls: Class type of the instance to release
            instance_id: ID of the instance to release (None for default group)
        """
        with cls._lock:
            if instance_cls in cls._registry:
                class_registry = cls._registry[instance_cls]
                if instance_id in class_registry:
                    del class_registry[instance_id]

    @classmethod
    def release_all_instances(
            cls,
            instance_cls: Optional[Type] = None
    ) -> None:
        """
        Release all instances of a specific class or all classes.

        Args:
            instance_cls: Class type to release (None releases all classes)
        """
        with cls._lock:
            if instance_cls:
                if instance_cls in cls._registry:
                    del cls._registry[instance_cls]
            else:
                cls._registry.clear()

    @classmethod
    def has_instance(
            cls,
            instance_cls: Type,
            instance_id: Optional[Hashable] = None
    ) -> bool:
        """
        Check if an instance exists in the registry.

        Args:
            instance_cls: Class type to check
            instance_id: ID of the instance to check

        Returns:
            True if instance exists, False otherwise
        """
        with cls._lock:
            return (
                    instance_cls in cls._registry and
                    instance_id in cls._registry[instance_cls]
            )

    @classmethod
    def get_all_instances(
            cls,
            instance_cls: Optional[Type] = None
    ) -> Dict:
        """
        Get a snapshot of current instances in the registry.

        Args:
            instance_cls: Class type to retrieve (None returns all classes)

        Returns:
            Copy of the registry snapshot
        """
        with cls._lock:
            if instance_cls:
                return cls._registry.get(instance_cls, {}).copy()
            return {
                cls_type: instances.copy()
                for cls_type, instances in cls._registry.items()
            }
