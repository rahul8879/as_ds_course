
database = {}
def create_user(username, email):
    if username not in database:
        database[username] = email
        return True
    return 'User already exists'

def read_user(username):
    return database.get(username, None)

def update_user(username, email):
    if username in database:
        database[username] = email
        return True
    return False

def delete_user(username):
    if username in database:
        del database[username]
        return True
    return False