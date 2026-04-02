"""/app/api/deps.py"""

from typing import Annotated, Any

from core.config import get_setting
from core.security import get_current_user, verify_master_key
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader
from infra.database.database import get_db
from service.model.user import UserBase, UserRoles

settings = get_setting()

api_key_header = APIKeyHeader(name="Authorization")

# --- Database ---
DatabaseDep = Annotated[Any, Depends(get_db)]

# --- User Data ---
CurrentUserDep = Annotated[UserBase, Depends(get_current_user)]

# --- Access control ---
default_admin_roles = [
    UserRoles.admin,
    UserRoles.superadmin,
]
superadmin_roles = [UserRoles.superadmin]


class AuthenticateWithRole:
    def __init__(self, role_for_grant=default_admin_roles, check_orgs=True):
        self.role_for_grant = role_for_grant
        self.check_orgs = check_orgs

    def __call__(self, current_user: CurrentUserDep):
        user_role = current_user["role"]
        if user_role.default in self.role_for_grant:
            return current_user
        if self.check_orgs and any(org.role in self.role_for_grant for org in user_role.organizations):
            return current_user
        raise HTTPException(status_code=403, detail="The user doesn't have enough privileges")


# --- Admin ---
authenticator = AuthenticateWithRole()
AdminDep = Annotated[str, Depends(authenticator)]

# --- Super Admin ---
super_authenticator = AuthenticateWithRole(superadmin_roles, check_orgs=False)
SuperAdminDep = Annotated[str, Depends(AuthenticateWithRole(superadmin_roles, check_orgs=False))]

# --- System Admin ---
SystemAdminDep = Annotated[str, Depends(verify_master_key)]
