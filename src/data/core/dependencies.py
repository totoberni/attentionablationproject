import pkg_resources
import subprocess
from typing import Dict, List, Optional, Set
import logging
import os
from pathlib import Path
from packaging import version
from .base import BaseManager, ConfigurationManager

class DependencyManager(BaseManager):
    """Manages Python package dependencies and their installation."""
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        requirements_file: Optional[str] = None
    ):
        super().__init__("DependencyManager")
        self.config_manager = config_manager
        
        # Get requirements file from config or use default
        if requirements_file is None:
            config = self.config_manager.get_config("model_config")
            requirements_file = config.get("requirements_file", "docker/requirements.txt")
        
        self.requirements_file = Path(requirements_file)
        self._installed_packages = self._get_installed_packages()
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """Get dictionary of installed packages and their versions."""
        return {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all required dependencies are installed with correct versions."""
        if not self.requirements_file.exists():
            raise FileNotFoundError(f"Requirements file not found: {self.requirements_file}")
        
        status = {}
        with open(self.requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    pkg_name, required_version = self._parse_requirement(line)
                    installed_version = self._installed_packages.get(pkg_name)
                    
                    if installed_version:
                        status[pkg_name] = self._check_version(
                            installed_version,
                            required_version
                        )
                    else:
                        status[pkg_name] = False
                        self.logger.warning(f"Package not installed: {pkg_name}")
        
        return status
    
    def install_dependencies(self, missing_only: bool = True) -> bool:
        """Install required dependencies."""
        try:
            if missing_only:
                status = self.check_dependencies()
                to_install = [pkg for pkg, installed in status.items() if not installed]
            else:
                with open(self.requirements_file, 'r') as f:
                    to_install = [
                        line.strip() for line in f 
                        if line.strip() and not line.startswith('#')
                    ]
            
            if to_install:
                self.logger.info(f"Installing packages: {', '.join(to_install)}")
                subprocess.check_call([
                    "pip", "install", "--no-cache-dir", *to_install
                ])
                self._installed_packages = self._get_installed_packages()
                self.logger.info("Package installation completed successfully")
            else:
                self.logger.info("All required packages are already installed")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {str(e)}")
            return False
    
    def _parse_requirement(self, req_line: str) -> tuple[str, str]:
        """Parse package name and version from requirement line."""
        import re
        
        # Handle comments and whitespace
        req_line = req_line.split('#')[0].strip()
        if not req_line:
            return '', ''
        
        # Match version specifiers
        version_pattern = r'(>=|==|<=|!=|~=|>|<)'
        parts = re.split(version_pattern, req_line, maxsplit=1)
        
        pkg_name = parts[0].strip()
        if len(parts) > 1:
            version_spec = parts[1] + parts[2].strip()
        else:
            version_spec = ''
        
        return pkg_name, version_spec
    
    def _check_version(self, installed: str, required: str) -> bool:
        """Check if installed version meets requirement using packaging.version."""
        if not required:
            return True
        
        try:
            installed_ver = version.parse(installed)
            
            # Handle different version specifiers
            import re
            match = re.match(r'([<>=!~]+)(.*)', required)
            if not match:
                return True
            
            operator, required_ver = match.groups()
            required_ver = version.parse(required_ver)
            
            if operator == '>=':
                return installed_ver >= required_ver
            elif operator == '==':
                return installed_ver == required_ver
            elif operator == '<=':
                return installed_ver <= required_ver
            elif operator == '!=':
                return installed_ver != required_ver
            elif operator == '~=':
                # Compatible release operator
                return installed_ver >= required_ver and installed_ver.release[0] == required_ver.release[0]
            elif operator == '>':
                return installed_ver > required_ver
            elif operator == '<':
                return installed_ver < required_ver
            
            return False
            
        except Exception as e:
            self.logger.warning(
                f"Could not compare versions: {installed} vs {required} - {str(e)}"
            )
            return True
    
    def get_missing_dependencies(self) -> Set[str]:
        """Get set of missing dependencies."""
        status = self.check_dependencies()
        missing = {pkg for pkg, installed in status.items() if not installed}
        if missing:
            self.logger.warning(f"Missing dependencies: {', '.join(missing)}")
        return missing
    
    def export_dependencies(self, output_file: Optional[str] = None) -> None:
        """Export currently installed dependencies to requirements file."""
        if output_file is None:
            output_file = self.requirements_file
        
        with open(output_file, 'w') as f:
            for pkg, ver in sorted(self._installed_packages.items()):
                f.write(f"{pkg}>={ver}\n")
        
        self.logger.info(f"Exported dependencies to: {output_file}") 