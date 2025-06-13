# src/core/utils/config_loader.py
"""
Configuration loader utility for OCR and other services.
Handles loading and validation of YAML configuration files.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Utility class for loading and validating configuration files."""
    
    @staticmethod
    def load_ocr_config(config_path: str = "src/configs/ocr_config.yaml") -> Dict[str, Any]:
        """
        Load and validate OCR configuration from YAML file.
        
        Args:
            config_path: Path to the OCR configuration file
            
        Returns:
            Dictionary containing OCR configuration with validated settings
        """
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                logger.warning(f"OCR config file not found: {config_path}")
                logger.info("Creating default configuration...")
                default_config = ConfigLoader._get_default_ocr_config()
                ConfigLoader._create_default_config_file(config_path, default_config)
                return default_config
            
            with open(config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            # Validate and normalize configuration
            validated_config = ConfigLoader._validate_ocr_config(config)
            
            logger.info(f"Successfully loaded OCR configuration from {config_path}")
            return validated_config
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {config_path}: {e}")
            logger.info("Using default configuration due to parsing error")
            return ConfigLoader._get_default_ocr_config()
        except Exception as e:
            logger.error(f"Error loading OCR config from {config_path}: {e}")
            logger.info("Using default configuration due to loading error")
            return ConfigLoader._get_default_ocr_config()
    
    @staticmethod
    def _validate_ocr_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize OCR configuration."""
        if not isinstance(config, dict):
            logger.error("Configuration is not a valid dictionary")
            return ConfigLoader._get_default_ocr_config()
        
        default_config = ConfigLoader._get_default_ocr_config()
        
        # Merge with defaults to ensure all required keys exist
        for section in default_config:
            if section not in config:
                config[section] = default_config[section]
                logger.warning(f"Missing config section '{section}', using defaults")
            else:
                # Ensure section is a dictionary
                if not isinstance(config[section], dict):
                    logger.error(f"Config section '{section}' is not a dictionary, using defaults")
                    config[section] = default_config[section]
                    continue
                
                # Merge section-specific defaults
                for key in default_config[section]:
                    if key not in config[section]:
                        config[section][key] = default_config[section][key]
                        logger.warning(f"Missing config key '{section}.{key}', using default: {default_config[section][key]}")
        
        # Validate specific configuration values
        ConfigLoader._validate_config_values(config)
        
        # Validate paths exist if specified
        ConfigLoader._validate_paths(config)
        
        return config
    
    @staticmethod
    def _validate_config_values(config: Dict[str, Any]) -> None:
        """Validate specific configuration values."""
        # Validate Tesseract settings
        tesseract_config = config.get('tesseract', {})
        
        # Validate PSM (Page Segmentation Mode)
        psm = tesseract_config.get('psm')
        if psm is not None and not (0 <= psm <= 13):
            logger.warning(f"Invalid Tesseract PSM value: {psm}, using default: 4")
            tesseract_config['psm'] = 4
        
        # Validate OEM (OCR Engine Mode)
        oem = tesseract_config.get('oem')
        if oem is not None and not (0 <= oem <= 3):
            logger.warning(f"Invalid Tesseract OEM value: {oem}, using default: 3")
            tesseract_config['oem'] = 3
        
        # Validate language string
        lang = tesseract_config.get('lang')
        if lang and not isinstance(lang, str):
            logger.warning(f"Invalid Tesseract language format: {lang}, using default: 'eng+kor'")
            tesseract_config['lang'] = 'eng+kor'
        
        # Validate EasyOCR settings
        easyocr_config = config.get('easyocr', {})
        
        # Validate languages list
        langs = easyocr_config.get('langs')
        if langs and not isinstance(langs, list):
            logger.warning(f"Invalid EasyOCR languages format: {langs}, using default: ['en', 'ko']")
            easyocr_config['langs'] = ['en', 'ko']
        
        # Validate GPU setting
        gpu = easyocr_config.get('gpu')
        if gpu is not None and not isinstance(gpu, bool):
            logger.warning(f"Invalid EasyOCR GPU setting: {gpu}, using default: True")
            easyocr_config['gpu'] = True
    
    @staticmethod
    def _validate_paths(config: Dict[str, Any]) -> None:
        """Validate that specified paths exist and are accessible."""
        path_configs = [
            ('tesseract', 'path', 'Tesseract executable'),
            ('poppler', 'path', 'Poppler binaries directory'),
            ('libreoffice', 'path', 'LibreOffice executable')
        ]
        
        for section, key, description in path_configs:
            if section in config and key in config[section]:
                path = config[section][key]
                if path:  # Only validate if path is not None or empty
                    path_obj = Path(path)
                    if not path_obj.exists():
                        logger.warning(f"{description} path does not exist: {path}")
                        logger.info(f"Please verify {section}.{key} in your configuration")
                    else:
                        logger.debug(f"Validated {description} path: {path}")
    
    @staticmethod
    def _get_default_ocr_config() -> Dict[str, Any]:
        """Get default OCR configuration with sensible defaults."""
        return {
            'tesseract': {
                'path': None,  # Use system PATH
                'lang': 'eng+kor',  # English + Korean languages
                'psm': 4,  # Assume single column of text of variable sizes
                'oem': 3   # Default, based on what is available (LSTM + Legacy)
            },
            'easyocr': {
                'langs': ['en', 'ko'],  # English and Korean
                'gpu': True  # Use GPU if available
            },
            'poppler': {
                'path': None  # Use system PATH
            },
            'libreoffice': {
                'path': None  # Use system PATH
            }
        }
    
    @staticmethod
    def _create_default_config_file(config_path: str, config: Dict[str, Any]) -> None:
        """Create a default configuration file if it doesn't exist."""
        try:
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, allow_unicode=True, indent=2)
            
            logger.info(f"Created default OCR configuration file: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to create default config file {config_path}: {e}")
    
    @staticmethod
    def get_config_summary(config: Dict[str, Any]) -> str:
        """Get a human-readable summary of the configuration."""
        summary_parts = []
        
        # Tesseract info
        tesseract_info = config.get('tesseract', {})
        tesseract_path = tesseract_info.get('path') or 'System PATH'
        tesseract_lang = tesseract_info.get('lang', 'eng+kor')
        tesseract_psm = tesseract_info.get('psm', 4)
        tesseract_oem = tesseract_info.get('oem', 3)
        summary_parts.append(
            f"Tesseract: {tesseract_path} (Languages: {tesseract_lang}, PSM: {tesseract_psm}, OEM: {tesseract_oem})"
        )
        
        # EasyOCR info
        easyocr_info = config.get('easyocr', {})
        easyocr_langs = easyocr_info.get('langs', ['en', 'ko'])
        easyocr_gpu = easyocr_info.get('gpu', True)
        gpu_status = "GPU enabled" if easyocr_gpu else "CPU only"
        summary_parts.append(f"EasyOCR: {', '.join(easyocr_langs)} ({gpu_status})")
        
        # Poppler info
        poppler_info = config.get('poppler', {})
        poppler_path = poppler_info.get('path') or 'System PATH'
        summary_parts.append(f"Poppler: {poppler_path}")
        
        # LibreOffice info
        libreoffice_info = config.get('libreoffice', {})
        libreoffice_path = libreoffice_info.get('path') or 'System PATH'
        summary_parts.append(f"LibreOffice: {libreoffice_path}")
        
        return "\n".join(summary_parts)
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str = "src/configs/ocr_config.yaml") -> bool:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary to save
            config_path: Path where to save the configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, allow_unicode=True, indent=2)
            
            logger.info(f"Configuration saved to: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            return False
    
    @staticmethod
    def validate_config_file(config_path: str = "src/configs/ocr_config.yaml") -> Dict[str, Any]:
        """
        Validate existing configuration file and return validation results.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'config': None
        }
        
        try:
            config = ConfigLoader.load_ocr_config(config_path)
            validation_results['config'] = config
            
            # Check if all paths are valid
            path_configs = [
                ('tesseract', 'path'),
                ('poppler', 'path'),  
                ('libreoffice', 'path')
            ]
            
            for section, key in path_configs:
                if section in config and key in config[section]:
                    path = config[section][key]
                    if path and not Path(path).exists():
                        validation_results['warnings'].append(
                            f"{section}.{key} path does not exist: {path}"
                        )
            
            logger.info("Configuration validation completed")
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(str(e))
            logger.error(f"Configuration validation failed: {e}")
        
        return validation_results

# Convenience functions for common operations
def load_ocr_config(config_path: str = "src/configs/ocr_config.yaml") -> Dict[str, Any]:
    """Convenience function to load OCR configuration."""
    return ConfigLoader.load_ocr_config(config_path)

def get_config_summary(config: Dict[str, Any]) -> str:
    """Convenience function to get configuration summary."""
    return ConfigLoader.get_config_summary(config)

def validate_config(config_path: str = "src/configs/ocr_config.yaml") -> Dict[str, Any]:
    """Convenience function to validate configuration."""
    return ConfigLoader.validate_config_file(config_path)

# Example usage and testing
if __name__ == "__main__":
    # Test the configuration loader
    print("Testing OCR Configuration Loader...")
    
    try:
        # Load configuration
        config = load_ocr_config()
        print("\nConfiguration loaded successfully!")
        
        # Print summary
        summary = get_config_summary(config)
        print(f"\nConfiguration Summary:\n{summary}")
        
        # Validate configuration
        validation = validate_config()
        print(f"\nValidation Results:")
        print(f"Valid: {validation['valid']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")
        if validation['errors']:
            print(f"Errors: {validation['errors']}")
            
    except Exception as e:
        print(f"Error testing configuration loader: {e}")