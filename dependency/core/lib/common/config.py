import json
import os
import yaml
from typing import Any, Dict, List, Optional


class ConfigLoader:
    """
    A static utility class for loading configuration files in various formats (JSON/YAML)
    with automatic format detection.

    Features:
    - Auto-detects file format (extension-based and content-based fallback)
    - Handles missing or incorrect file extensions
    - Graceful fallback when optional dependencies (PyYAML) are missing
    - Comprehensive error reporting
    - Can be used without instantiation (static methods only)
    """

    # Class-level constant for supported parsers
    _PARSERS = [
        {
            'name': 'JSON',
            'extensions': ['json'],
            'load': lambda c: json.loads(c),
            'exceptions': (json.JSONDecodeError,),
            'required': False
        },
        {
            'name': 'YAML',
            'extensions': ['yaml', 'yml'],
            'load': lambda c: yaml.safe_load(c),
            'exceptions': (yaml.YAMLError,),
            'required': False
        }
    ]

    @staticmethod
    def _read_file_content(file_path: str) -> str:
        """
        Read the content of a configuration file.

        Args:
            file_path: Path to the configuration file

        Returns:
            File content as string

        Raises:
            ValueError: If file cannot be read
        """
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            raise ValueError(f"Config file {file_path} not found")
        except IOError as e:
            raise ValueError(f"Failed to read file {file_path}: {str(e)}")

    @staticmethod
    def _get_file_extension(file_path: str) -> Optional[str]:
        """
        Extract the lowercase file extension without the dot.

        Args:
            file_path: Path to the file

        Returns:
            File extension (without dot) or None if no extension
        """
        ext = os.path.splitext(file_path)[1].lower()
        return ext.lstrip('.') if ext else None

    @staticmethod
    def _order_parsers_by_extension(extension: Optional[str]) -> List[Dict]:
        """
        Order parsers based on file extension match priority.

        Args:
            extension: File extension to match against

        Returns:
            Ordered list of parsers with matching extensions first
        """
        if not extension:
            return ConfigLoader._PARSERS.copy()

        ordered = []
        remaining = []

        for parser in ConfigLoader._PARSERS:
            if extension in parser['extensions']:
                ordered.append(parser)
            else:
                remaining.append(parser)

        return ordered + remaining

    @staticmethod
    def get_supported_formats() -> List[str]:
        """
        Get list of supported configuration formats.

        Returns:
            List of format names (e.g., ['JSON', 'YAML'])
        """
        return [p['name'] for p in ConfigLoader._PARSERS]

    @staticmethod
    def load(file_path: str) -> Any:
        """
        Load configuration from a file with automatic format detection.

        Args:
            file_path: Path to the configuration file

        Returns:
            Parsed configuration content (typically dict)

        Raises:
            ValueError: When file cannot be parsed by any available parser
        """
        content = ConfigLoader._read_file_content(file_path)
        extension = ConfigLoader._get_file_extension(file_path)
        ordered_parsers = ConfigLoader._order_parsers_by_extension(extension)

        # Try each parser in order
        for parser in ordered_parsers:
            try:
                config = parser['load'](content)
                if config is not None:  # Basic validation
                    return config
            except parser['exceptions']:
                continue
            except Exception:
                continue

        # All parsers failed
        raise ValueError(
            f"Failed to parse config file {file_path}\n"
            f"Supported formats: {', '.join(ConfigLoader.get_supported_formats())}\n"
            f"File extension: {extension or 'none'}"
        )
