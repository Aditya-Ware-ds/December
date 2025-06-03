"""
Configuration Manager for Pose Estimation System

This module handles loading, merging, and validating configuration files
for the pose estimation system. It supports YAML configuration files
with hierarchical overrides and environment variable substitution.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from copy import deepcopy
import re


class ConfigManager:
    """
    Configuration manager for pose estimation system.

    Handles loading default configurations, user overrides, and environment
    variable substitution with comprehensive validation.
    """

    def __init__(self, default_config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            default_config_path: Path to default configuration file
        """
        self.logger = self._setup_logger()

        if default_config_path is None:

            current_dir = Path(__file__).parent
            default_config_path = current_dir / "default_config.yaml"

        self.default_config_path = Path(default_config_path)
        self.config = {}
        self.config_sources = []

        self._load_default_config()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for configuration manager"""
        logger = logging.getLogger(f"{__name__}.ConfigManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _load_default_config(self):
        """Load the default configuration file"""
        try:
            if self.default_config_path.exists():
                with open(self.default_config_path, "r") as f:
                    self.config = yaml.safe_load(f)
                self.config_sources.append(str(self.default_config_path))
                self.logger.info(
                    f"Loaded default config from: {self.default_config_path}"
                )
            else:
                self.logger.warning(
                    f"Default config file not found: {self.default_config_path}"
                )
                self.config = self._get_minimal_default_config()
                self.config_sources.append("minimal_default")
        except Exception as e:
            self.logger.error(f"Error loading default config: {str(e)}")
            self.config = self._get_minimal_default_config()
            self.config_sources.append("minimal_default")

    def _get_minimal_default_config(self) -> Dict[str, Any]:
        """Return minimal default configuration when file is not available"""
        return {
            "model": {
                "name": "mmpose",
                "device": "auto",
                "mmpose": {"checkpoint": "td-hm_hrnet-w32_8xb64-210e_coco-256x192"},
            },
            "processing": {
                "batch_size": 32,
                "confidence_threshold": 0.3,
                "frame_processing": {"frame_skip": 1},
            },
            "output": {
                "format": "json",
                "json": {"indent": 2, "include_metadata": True},
            },
            "logging": {"level": "INFO", "progress_bar": {"enabled": True}},
        }

    def load_config(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        override_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Load and merge configuration from multiple sources.

        Args:
            config_path: Path to user configuration file
            config_dict: Configuration dictionary to merge
            override_dict: Dictionary of overrides to apply

        Returns:
            Merged configuration dictionary
        """

        merged_config = deepcopy(self.config)

        if config_path:
            user_config = self._load_yaml_file(config_path)
            if user_config:
                merged_config = self._deep_merge(merged_config, user_config)
                self.config_sources.append(config_path)

        if config_dict:
            merged_config = self._deep_merge(merged_config, config_dict)
            self.config_sources.append("config_dict")

        if override_dict:
            merged_config = self._deep_merge(merged_config, override_dict)
            self.config_sources.append("override_dict")

        merged_config = self._substitute_env_vars(merged_config)

        validation_errors = self.validate_config(merged_config)
        if validation_errors:
            self.logger.warning("Configuration validation issues found:")
            for error in validation_errors:
                self.logger.warning(f"  - {error}")

        merged_config = self._post_process_config(merged_config)

        self.config = merged_config
        return merged_config

    def _load_yaml_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load YAML file and return dictionary"""
        try:
            path = Path(file_path)
            if not path.exists():
                self.logger.error(f"Config file not found: {file_path}")
                return None

            with open(path, "r") as f:
                config = yaml.safe_load(f)

            self.logger.info(f"Loaded config from: {file_path}")
            return config

        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error in {file_path}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading config file {file_path}: {str(e)}")
            return None

    def _deep_merge(
        self, base_dict: Dict[str, Any], override_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with override_dict taking precedence.

        Args:
            base_dict: Base dictionary
            override_dict: Dictionary to merge (takes precedence)

        Returns:
            Merged dictionary
        """
        result = deepcopy(base_dict)

        for key, value in override_dict.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration values.

        Environment variables should be specified as ${VAR_NAME} or ${VAR_NAME:default_value}

        Args:
            config: Configuration dictionary

        Returns:
            Configuration with environment variables substituted
        """

        def substitute_value(value):
            if isinstance(value, str):

                pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

                def replace_env_var(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ""
                    return os.getenv(var_name, default_value)

                return re.sub(pattern, replace_env_var, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value

        return substitute_value(config)

    def _post_process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply post-processing to configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Post-processed configuration
        """

        config = self._convert_string_booleans(config)

        self._setup_logging_from_config(config)

        config = self._resolve_paths(config)

        return config

    def _convert_string_booleans(self, obj: Any) -> Any:
        """Convert string boolean values to actual booleans"""
        if isinstance(obj, dict):
            return {k: self._convert_string_booleans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_string_booleans(item) for item in obj]
        elif isinstance(obj, str):
            if obj.lower() in ("true", "yes", "1"):
                return True
            elif obj.lower() in ("false", "no", "0"):
                return False
            else:
                return obj
        else:
            return obj

    def _setup_logging_from_config(self, config: Dict[str, Any]):
        """Setup logging based on configuration"""
        logging_config = config.get("logging", {})

        log_level = logging_config.get("level", "INFO")
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)

        logging.getLogger().setLevel(numeric_level)
        self.logger.setLevel(numeric_level)

    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve relative paths in configuration"""

        def resolve_path_value(value, key):
            if isinstance(value, str) and key in [
                "path",
                "output_dir",
                "checkpoint_path",
            ]:
                if not os.path.isabs(value):

                    return os.path.abspath(value)
            return value

        def resolve_dict(d):
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = resolve_dict(v)
                elif isinstance(v, list):
                    result[k] = [
                        resolve_path_value(item, k) if isinstance(item, str) else item
                        for item in v
                    ]
                else:
                    result[k] = resolve_path_value(v, k)
            return result

        return resolve_dict(config)

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration and return list of errors.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation error messages
        """
        errors = []

        required_sections = ["model", "processing", "output"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")

        model_config = config.get("model", {})
        if "name" not in model_config:
            errors.append("Missing model.name in configuration")

        processing_config = config.get("processing", {})

        if "confidence_threshold" in processing_config:
            threshold = processing_config["confidence_threshold"]
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                errors.append("processing.confidence_threshold must be between 0 and 1")

        if "batch_size" in processing_config:
            batch_size = processing_config["batch_size"]
            if not isinstance(batch_size, int) or batch_size < 1:
                errors.append("processing.batch_size must be a positive integer")

        device = model_config.get("device", "auto")
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if device not in valid_devices:
            errors.append(f"model.device must be one of: {valid_devices}")

        return errors

    def save_config(self, output_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Save current configuration to YAML file.

        Args:
            output_path: Path to save configuration
            config: Configuration to save (uses current if None)
        """
        config_to_save = config or self.config

        try:
            with open(output_path, "w") as f:
                yaml.dump(config_to_save, f, default_flow_style=False, indent=2)
            self.logger.info(f"Configuration saved to: {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")

    def get_model_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model-specific configuration.

        Args:
            model_name: Name of the model (uses current if None)

        Returns:
            Model-specific configuration dictionary
        """
        if model_name is None:
            model_name = self.config.get("model", {}).get("name", "mmpose")

        model_config = self.config.get("model", {}).copy()

        if model_name in model_config:
            model_specific = model_config[model_name]

            for key, value in model_specific.items():
                model_config[key] = value

        return model_config

    def print_config_summary(self):
        """Print a summary of the current configuration"""
        print("\n" + "=" * 50)
        print("CONFIGURATION SUMMARY")
        print("=" * 50)

        print(f"Config sources: {', '.join(self.config_sources)}")

        model_config = self.config.get("model", {})
        print(f"Model: {model_config.get('name', 'Unknown')}")
        print(f"Device: {model_config.get('device', 'auto')}")

        processing_config = self.config.get("processing", {})
        print(f"Batch size: {processing_config.get('batch_size', 'Unknown')}")
        print(
            f"Confidence threshold: {processing_config.get('confidence_threshold', 'Unknown')}"
        )

        output_config = self.config.get("output", {})
        print(f"Output format: {output_config.get('format', 'Unknown')}")

        print("=" * 50 + "\n")


def load_config(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    override_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to load configuration.

    Args:
        config_path: Path to user configuration file
        config_dict: Configuration dictionary to merge
        override_dict: Dictionary of overrides to apply

    Returns:
        Merged configuration dictionary
    """
    manager = ConfigManager()
    return manager.load_config(config_path, config_dict, override_dict)


def create_config_template(output_path: str, model_name: str = "mmpose"):
    """
    Create a configuration template file.

    Args:
        output_path: Path to save template
        model_name: Model name for template
    """
    template = {
        "model": {"name": model_name, "device": "auto"},
        "processing": {
            "batch_size": 16,
            "confidence_threshold": 0.5,
            "frame_processing": {"frame_skip": 1},
        },
        "output": {"format": "json", "visualization": {"enabled": False}},
    }

    try:
        with open(output_path, "w") as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        print(f"Configuration template saved to: {output_path}")
    except Exception as e:
        print(f"Error saving template: {str(e)}")


if __name__ == "__main__":

    manager = ConfigManager()

    config = manager.load_config(
        override_dict={"processing": {"batch_size": 16, "confidence_threshold": 0.5}}
    )

    manager.print_config_summary()

    manager.save_config("current_config.yaml")

    create_config_template("config_template.yaml", "mmpose")
