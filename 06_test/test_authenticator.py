import pytest
from authenticator import Authenticator

#1:register()メソッドでユーザーが正しく登録されるか
def test_register_success():
    auth = Authenticator()
    auth.register("user1","password123")
    assert auth.users["user1"] == "password123"

#2.register()メソッドですでに存在するユーザー名で登録を試みた場合にエラーが発生するか
def test_register_duplicate_user():
    auth = Authenticator()
    auth.register("user1","password123")
    with pytest.raises(ValueError):
        auth.register("user1","password456")

#3.login()メソッドで正しいユーザー名とパスワードでログインできるか
def test_login_success():
    auth = Authenticator()
    auth.register("user1","password123")
    result = auth.login("user1","password123")
    assert result == "ログイン成功"

#4.login()メソッドで誤ったパスワードでエラーが出るか
def test_login_wrong_password():
    auth = Authenticator()
    auth.register("user1","password123")
    with pytest.raises(ValueError):
        auth.login("user1","wrong_password")