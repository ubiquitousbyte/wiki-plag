from typing import Dict


def create_backend_conf(password: str = None) -> Dict[str, str]:
    """
    Creates a MongoDB backend configuration 

    Parameters
    ----------
    password : str = None
        The password to use to connect to the database
        If the password is None, this function assumes that a file 
        called /run/secrets/db-password exists and stores the password.
        The function will try and read this file. 
    """

    if password is None:
        with open("/run/secrets/db-password", 'r') as pwd_file:
            password = pwd_file.read()
    settings = {
        "user": "wikiplag",
        "password": password,
        "taskmeta_collection": "tasks",
    }
    return settings
