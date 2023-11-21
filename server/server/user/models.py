from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()

class User:
    def __init__(self, username, password, email):
        self.username = username
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')
        self.email = email
        
