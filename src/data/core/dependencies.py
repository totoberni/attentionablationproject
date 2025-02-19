import pkg_resources
import subprocess
from typing import Dict, List, Optional, Set
import logging
import os
from pathlib import Path

class DependencyManager:
    def __init__(self, requirements_file: str = "docker/requirements.txt"):
        self.requirements_file = Path(requirements_file)
        self.logger = logging.getLogger(__name__)
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
        
        return status
    
    def install_dependencies(self, missing_only: bool = True) -> bool:
        """Install required dependencies."""
        try:
            if missing_only:
                status = self.check_dependencies()
                to_install = [pkg for pkg, installed in status.items() if not installed]
            else:
                with open(self.requirements_file, 'r') as f:
                    to_install = [line.strip() for line in f if line.strip()]
            
            if to_install:
                self.logger.info(f"Installing packages: {', '.join(to_install)}")
                subprocess.check_call([
                    "pip", "install", "--no-cache-dir", *to_install
                ])
                self._installed_packages = self._get_installed_packages()
            
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to install dependencies: {str(e)}")
            return False
    
    def _parse_requirement(self, req_line: str) -> tuple[str, str]:
        """Parse package name and version from requirement line."""
        if '>=' in req_line:
            pkg_name, version = req_line.split('>=')
        elif '==' in req_line:
            pkg_name, version = req_line.split('==')
        else:
            pkg_name = req_line
            version = ''
        
        return pkg_name.strip(), version.strip()
    
    def _check_version(self, installed: str, required: str) -> bool:
        """Check if installed version meets requirement."""
        if not required:
            return True
        
        try:
            installed_parts = [int(x) for x in installed.split('.')]
            required_parts = [int(x) for x in required.split('.')]
            
            for i, r in zip(installed_parts, required_parts):
                if i < r:
                    return False
                elif i > r:
                    return True
            return len(installed_parts) >= len(required_parts)
        except ValueError:
            self.logger.warning(
                f"Could not compare versions: {installed} vs {required}"
            )
            return True
    
    def get_missing_dependencies(self) -> Set[str]:
        """Get set of missing dependencies."""
        status = self.check_dependencies()
        return {pkg for pkg, installed in status.items() if not installed}
    
    def export_dependencies(self, output_file: Optional[str] = None) -> None:
        """Export currently installed dependencies to requirements file."""
        if output_file is None:
            output_file = self.requirements_file
        
        with open(output_file, 'w') as f:
            for pkg, version in sorted(self._installed_packages.items()):
                f.write(f"{pkg}>={version}\n") 