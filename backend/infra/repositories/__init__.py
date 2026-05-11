from __future__ import annotations

"""兼容旧仓储包出口"""

from backend.infra.database import SysUserRepository, check_mysql_connectivity

check_db_connectivity = check_mysql_connectivity

__all__ = ["SysUserRepository", "check_db_connectivity"]
