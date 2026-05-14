import logging
from parse_logging.parse_logger import make_logger


def test_logger_writes_to_file(tmp_path):
    log_file = tmp_path / "parse.log"
    logger = make_logger("test", str(log_file), level=logging.DEBUG)
    logger.warning("test warning message")

    content = log_file.read_text()
    assert "test warning message" in content


def test_logger_has_console_handler(tmp_path):
    log_file = tmp_path / "parse.log"
    logger = make_logger("test2", str(log_file), level=logging.DEBUG)
    handlers = logger.handlers
    types = [type(h).__name__ for h in handlers]
    assert "StreamHandler" in types
    assert "FileHandler" in types


def test_logger_level_respected(tmp_path):
    log_file = tmp_path / "parse.log"
    logger = make_logger("test3", str(log_file), level=logging.WARNING)
    logger.debug("should not appear")
    logger.warning("should appear")

    content = log_file.read_text()
    assert "should not appear" not in content
    assert "should appear" in content
