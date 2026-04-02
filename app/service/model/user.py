"""/app/service/model/user.py"""

from pydantic import BaseModel, EmailStr


class UserRoles:
    common: str = "common"
    admin: str = "admin"
    superadmin: str = "superadmin"


class UserOrganizationInfo(BaseModel):
    org_id: int
    org_name: str
    org_description: str | None = None
    role: str | None = None


class UserOrganizationRole(BaseModel):
    default: str = "common"
    organizations: list[UserOrganizationInfo] = []


class UserBase(BaseModel):
    user_id: str
    email: EmailStr | None = None
    username: str
    department: str | None = None
    role: str
    company: str
