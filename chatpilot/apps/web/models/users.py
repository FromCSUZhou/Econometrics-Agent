import time
from typing import List, Optional

from loguru import logger
import peewee as pw
from playhouse.shortcuts import model_to_dict
from pydantic import BaseModel

from chatpilot.apps.db import DB
from chatpilot.apps.web.models.chats import Chats

import json


####################
# User DB Schema
####################


class User(pw.Model):
    id = pw.CharField(unique=True)
    name = pw.CharField()
    email = pw.CharField()
    role = pw.CharField()
    profile_image_url = pw.CharField()
    timestamp = pw.DateField()
    uploaded_files = pw.TextField(default='[]')  # 存储为JSON字符串
    quota = pw.IntegerField()

    class Meta:
        database = DB


class UserModel(BaseModel):
    id: str
    name: str
    email: str
    role: str = "pending"
    profile_image_url: str = "/user.png"
    timestamp: int  # timestamp in epoch
    uploaded_files: List[str] = []  # 存储文件名列表
    quota: int


####################
# Forms
####################


class UserRoleUpdateForm(BaseModel):
    id: str
    role: str


class UserUpdateForm(BaseModel):
    name: str
    email: str
    profile_image_url: str
    password: Optional[str] = None


class UsersTable:
    def __init__(self, db):
        self.db = db
        self.db.create_tables([User])

    def insert_new_user(
            self, id: str, name: str, email: str, role: str = "pending", quota: int = 50
    ) -> Optional[UserModel]:
        # default user quota 50
        user = UserModel(
            **{
                "id": id,
                "name": name,
                "email": email,
                "role": role,
                "profile_image_url": "/user.png",
                "timestamp": int(time.time()),
                "uploaded_files": [],
                "quota": quota,
            }
        )
        result = User.create(**user.dict())
        if result:
            return user
        else:
            return None

    def get_user_by_id(self, id: str) -> Optional[UserModel]:
        # try:
            user = User.get(User.id == id)
            user_dict = model_to_dict(user)
            # transform the JSON string to a Python list
            user_dict['uploaded_files'] = json.loads(user_dict.get('uploaded_files', '[]'))
            return UserModel(**user_dict)
        # except Exception as e:
        #     logger.error(f"Error getting user by id: {e}")
        #     return None

    def get_user_by_email(self, email: str) -> Optional[UserModel]:
        try:
            user = User.get(User.email == email)
            user_dict = model_to_dict(user)
            # transform the JSON string to a Python list
            user_dict['uploaded_files'] = json.loads(user_dict.get('uploaded_files', '[]'))
            return UserModel(**user_dict)
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None

    def get_users(self, skip: int = 0, limit: int = 50) -> List[UserModel]:
        try:
            return [
                UserModel(**{
                    **model_to_dict(user),
                    'uploaded_files': json.loads(model_to_dict(user).get('uploaded_files', '[]'))
                })
                for user in User.select()
            ]
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return []

    def get_num_users(self) -> Optional[int]:
        return User.select().count()

    def update_user_role_by_id(self, id: str, role: str) -> Optional[UserModel]:
        try:
            query = User.update(role=role).where(User.id == id)
            query.execute()

            user = User.get(User.id == id)
            return UserModel(**model_to_dict(user))
        except:
            return None

    def update_user_profile_image_url_by_id(
            self, id: str, profile_image_url: str
    ) -> Optional[UserModel]:
        try:
            query = User.update(profile_image_url=profile_image_url).where(
                User.id == id
            )
            query.execute()

            user = User.get(User.id == id)
            return UserModel(**model_to_dict(user))
        except:
            return None

    def update_user_by_id(self, id: str, updated: dict) -> Optional[UserModel]:
        try:
            query = User.update(**updated).where(User.id == id)
            query.execute()

            user = User.get(User.id == id)
            return UserModel(**model_to_dict(user))
        except:
            return None

    def delete_user_by_id(self, id: str) -> bool:
        try:
            # Delete User Chats
            result = Chats.delete_chats_by_user_id(id)

            if result:
                # Delete User
                query = User.delete().where(User.id == id)
                query.execute()  # Remove the rows, return number of rows removed.

                return True
            else:
                return False
        except:
            return False


Users = UsersTable(DB)
