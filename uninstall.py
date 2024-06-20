import subprocess
import pkg_resources

# Obtener todos los paquetes instalados
installed_packages = [pkg.key for pkg in pkg_resources.working_set]

# Desinstalar cada paquete
for package in installed_packages:
    subprocess.check_call(["pip", "uninstall", "-y", package])
