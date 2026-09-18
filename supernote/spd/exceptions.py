"""Library-specific exception classes for .spd drawing file parsing."""


class SpdError(Exception):
    """Base class for .spd parsing/rendering errors."""


class UnsupportedSpdFormat(SpdError):
    """Raised when the .spd file's format version is missing or unsupported."""


class CorruptSpdFile(SpdError):
    """Raised when the file is not a valid SQLite database or is missing required tables."""
