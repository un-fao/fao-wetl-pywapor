import os
import json
import pywapor
import getpass
import sys
import requests
import cdsapi
from pywapor.general.logger import log, adjust_logger
from cryptography.fernet import Fernet
from pywapor.collect.product.LANDSAT import espa_api

PASSWORD_INSTRUCTIONS = {
"NASA": """> Used for `MODIS`, `SRTM` and `MERRA2` data.
> Create an account at https://urs.earthdata.nasa.gov.
> Make sure to accept the terms of use at "Applications > Authorized Apps > Approve More Applications":
  * NASA GESDISC DATA ARCHIVE
  * LP DAAC OPeNDAP""",

"TERRA": """> Used for `TERRA` (VITO:PROVA-V).
> Create an account at https://viewer.terrascope.be.""",

"ECMWF": """> Used for `ERA5` (old, please use `"CDS"` instead).
> Create an account at https://cds.climate.copernicus.eu.
  * On your profile page, scroll to the "API key" section.
  * Accept conditions when running `setup("ECMWF")` for the first time.""",

"CDS": """> Used for `ERA5`.
> Create an account at https://cds-beta.climate.copernicus.eu.
  * On your profile page, scroll to the "Personal Access Token" section.
  * Accept conditions when running `setup("CDS")` for the first time.""",

"EARTHEXPLORER": """> Used for `LANDSAT`.
> Create an account at https://earthexplorer.usgs.gov.""",

"COPERNICUS_DATA_SPACE": """> Used for `SENTINEL2` and `SENTINEL3`.
> Create an account at https://dataspace.copernicus.eu.""",

"VIIRSL1": """> Used for `VIIRSL1`
> Normally you do not need this account: it is
> only used when using the `requests.get` download method, 
> which is not the default!
> Create an account at https://ladsweb.modaps.eosdis.nasa.gov/.
> In the top right, press "Login > Generate Token".""",

"LSASAF": """> Used for `MSG_MDSSFTD` and `MSG_MDIDSSF`.
> Create an account at https://mokey.lsasvcs.ipma.pt/.""",
}

def ask_pw(account):
    log.info(f"--> Setting up `{account}` account.").add()
    instructions = PASSWORD_INSTRUCTIONS.get(account, "")
    for line in instructions.split("\n"):
        log.info(line)
    if account == "WAPOR" or account == "VIIRSL1":
        account_name = ""
        pwd = input(f"{account} API token: ")
    elif account == "ECMWF":
        account_name = 'https://cds.climate.copernicus.eu/api/v2'
        api_key_1 = input(f"{account} UID: ")
        api_key_2 = input(f"{account} CDS API key: ")
        pwd = f"{api_key_1}:{api_key_2}"
    elif account == "CDS":
        account_name = 'https://cds-beta.climate.copernicus.eu/api'
        pat = input(f"{account} Personal Access Token: ")
        pwd = f"{pat}"
    else:
        account_name = input(f"{account} username: ")
        pwd = getpass.getpass(f"{account} password: ")  
    log.sub()          
    return account_name, pwd

def setup(account):
    """Asks, saves and tests a username/password combination for `account`.

    Parameters
    ----------
    account : {"NASA" | "TERRA" | "ECMWF" | "CDS" | "COPERNICUS_DATA_SPACE" | "EARTHEXPLORER" | "VIIRSL1" | "LSASAF"}
        Which un/pw combination to store.
    """

    folder = os.path.dirname(os.path.realpath(pywapor.__path__[0]))
    filename = "secret.txt"
    filehandle = os.path.join(folder, filename)
    json_filehandle = os.path.join(folder, "keys.json")

    if not os.path.exists(filehandle):
        create_key()
    
    with open(filehandle,"r") as f:
        key = f.read()

    cipher_suite = Fernet(key.encode('utf-8'))

    if os.path.exists(json_filehandle):
        # open json file  
        with open(json_filehandle) as f:
            datastore = f.read()
        obj = json.loads(datastore)      
    else:
        obj = {}

    n = 1
    max_attempts = 3
    succes = False

    while n <= max_attempts and not succes:

        account_name, pwd = ask_pw(account)

        log.info(f"--> Testing {account} un/pw.")
        succes, error = {
            "NASA": nasa_account,
            "EARTHEXPLORER": earthexplorer_account,
            "TERRA": terra_account,
            "ECMWF": ecmwf_account,
            "CDS": cds_account,
            "COPERNICUS_DATA_SPACE": copernicus_data_space_account,
            "VIIRSL1": viirs_account,
            "LSASAF": lsasaf_account,
        }[account]((account_name, pwd))

        if succes:
            log.info(f"--> {account} un/pw working.")
            # Encrypting un/pw
            account_name_crypt = cipher_suite.encrypt(("%s" %account_name).encode('utf-8'))
            pwd_crypt = cipher_suite.encrypt(("%s" %pwd).encode('utf-8'))
            obj[account] = ([str(account_name_crypt.decode("utf-8")), str(pwd_crypt.decode("utf-8"))])
            # Save to disk
            with open(json_filehandle, 'w') as outfile:
                json.dump(obj, outfile)     
        else:
            log.warning(f"--> ({n}/{max_attempts}) {account} not working ({error}).")

        n += 1

    if not succes:
        sys.exit(f"Please fix your {account} account.") 

    return

def get(account):
    """Loads a required username/password.

    Parameters
    ----------
    account : {"NASA" | "TERRA" | "ECMWF" | "CDS" | "COPERNICUS_DATA_SPACE" | "EARTHEXPLORER" | "VIIRSL1 | "LSASAF"}
        Which un/pw combination to load.
    """

    folder = os.path.dirname(os.path.realpath(pywapor.__path__[0]))
    filename = "secret.txt"
    key_file = os.path.join(folder, filename)
    json_file = os.path.join(folder, "keys.json")
    
    if not os.path.exists(key_file):
        create_key()
        if os.path.exists(json_file):
            print("removing old/invalid json file")
            try:
                os.remove(json_file)
            except PermissionError:
                log.warning("--> Unable to remove existing `keys.json` file (PermissionError), please remove the file manually before proceeding.")
    
    if not os.path.exists(json_file):
        setup(account)

    f = open(key_file,"r")
    key = f.read()
    f.close()
    
    cipher_suite = Fernet(key.encode('utf-8'))    
    
    # open json file  
    with open(json_file) as f:
        datastore = f.read()
    obj = json.loads(datastore)      
    f.close()

    if account not in obj.keys():
        setup(account)
        obj = None 
        with open(json_file) as f:
            datastore = f.read()
        obj = json.loads(datastore)      
        f.close()       

    username_crypt, pwd_crypt = obj[account]
    username = cipher_suite.decrypt(username_crypt.encode("utf-8"))
    pwd = cipher_suite.decrypt(pwd_crypt.encode("utf-8"))  

    return (str(username.decode("utf-8")), str(pwd.decode("utf-8")))

def create_key():
    """Generates a key file.
    """
    folder = os.path.dirname(os.path.realpath(pywapor.__path__[0]))
    filename = "secret.txt"
    filehandle = os.path.join(folder, filename)

    if os.path.exists(filehandle):
        try:
            os.remove(filehandle)
        except PermissionError:
            log.warning("--> Unable to remove existing `secret.txt` file (PermissionError), please remove the file manually before proceeding.")
        
    with open(filehandle,"w+") as f:
        f.write(str(Fernet.generate_key().decode("utf-8")))

def lsasaf_account(user_pw):

    test_url = r"https://thredds.lsasvcs.ipma.pt/thredds/catalog/catalog.html"

    session = requests.Session()
    session.auth = user_pw
    x = session.get(test_url)

    if not x.ok:
        error = str(x.status_code)
        succes = False
    else:
        error = ""
        succes = True

    return succes, error


def terra_account(user_pw):
    """Check if the given or stored TERA username/password combination
    is correct. Accounts can be created on https://viewer.terrascope.be.

    Parameters
    ----------
    user_pw : tuple, optional
        ("username", "password") to check, if `None` will try to load the 
        password from the keychain, by default None.

    Returns
    -------
    bool
        True if the password works, otherwise False.
    """

    url = "https://sso.vgt.vito.be/auth/realms/terrascope/protocol/openid-connect/token"
    params = {
        "grant_type": "password",
        "client_id": "public",
        "username": user_pw[0],
        "password": user_pw[1],
    }
    headers = {'content-type': 'application/x-www-form-urlencoded'}

    x = requests.post(url, data = params, headers = headers)

    error = ""

    if not x.ok:
        error = eval(x.text)["error_description"]
        succes = False
    else:
        succes = True

    return succes, error

def nasa_account(user_pw):
    """Check if the given or stored NASA username/password combination is 
    correct. Accounts can be created on https://urs.earthdata.nasa.gov/users/new.

    Parameters
    ----------
    user_pw : tuple, optional
        ("username", "password") to check, if `None` will try to load the 
        password from the keychain, by default None.

    Returns
    -------
    bool
        True if the password works, otherwise False.
    """

    folder = os.path.dirname(os.path.realpath(pywapor.__path__[0]))
    test_url = r"https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T3NXGLC.5.12.4/1987/08/MERRA2_100.tavg3_2d_glc_Nx.19870801.nc4"
    test_file = os.path.join(folder, "nasa_test.nc4")

    if os.path.isfile(test_file):
        try:
            os.remove(test_file)
        except PermissionError:
            ...

    username, password = user_pw

    try:
        x = requests.get(test_url, allow_redirects = False)
    except requests.exceptions.ConnectionError as e:
        # TODO fix this...
        log.add().warning("> Test not working right now, assuming your account is valid.")
        log.warning("> If you want to enter a new password, run `pywapor.collect.accounts.setup('NASA')`.").sub()
        succes = True
        return succes, ""

    y = requests.get(x.headers['location'], auth = (username, password))

    error = ""

    if x.ok and y.ok:

        with open(test_file, 'w+b') as z:
            z.write(y.content)
    
        if os.path.isfile(test_file):
            statinfo = os.stat(test_file)
            succes = statinfo.st_size == 3963517
            if not succes:
                error = "please add 'NASA GESDISC DATA ARCHIVE' to 'Approved Applications'."
        else:
            succes = False

    else:
        error = "wrong username/password."
        succes = False
    
    if os.path.isfile(test_file):
        try:
            os.remove(test_file)
        except PermissionError:
            ...

    return succes, error

def viirs_account(user_pw):
    """Check if the given or stored VIIRSL1 token is 
    correct. Accounts can be created on https://ladsweb.modaps.eosdis.nasa.gov/.
    Then in the top right, press "Login" and "generate token".

    Parameters
    ----------
    user_pw : tuple, optional
        ("", "token") to check, if `None` will try to load the 
        password from the keychain, by default None.

    Returns
    -------
    bool
        True if the password works, otherwise False.
    """

    _, token = user_pw

    headers = {'Authorization': 'Bearer ' + token}
    test_url = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5110/CLDMSK_L2_VIIRS_SNPP/2023/060/CLDMSK_L2_VIIRS_SNPP.A2023060.0006.001.2023233192108.nc"
    response = requests.Session().get(test_url, headers=headers, stream = True)
    for data in response.iter_content(chunk_size=1024):
        succes = b'DOCTYPE' not in data
        if not succes:
            error = "wrong token."
        else:
            error = ""
        break

    return succes, error


def earthexplorer_account(user_pw):
    """Check if the given or stored WAPOR token is 
    correct. Accounts can be created on https://wapor.apps.fao.org/home/WAPOR_2/1.

    Parameters
    ----------
    user_pw : tuple, optional
        ("", "token") to check, if `None` will try to load the 
        password from the keychain, by default None.

    Returns
    -------
    bool
        True if the password works, otherwise False.
    """

    username, pw = user_pw

    try:
        response = espa_api('user', uauth = (username, pw))
    except requests.exceptions.ConnectionError as e:
        # TODO fix this...
        log.add().warning("> Test not working right now, assuming your account is valid.")
        log.warning("> If you want to enter a new password, run `pywapor.collect.accounts.setup('EARTHEXPLORER')`.").sub()
        succes = True
        return succes, ""

    if isinstance(response, type(None)):
        response = {"username": "x"}

    if "email" in response.keys() and response["username"] == username:
        succes = True
        error = ""
    elif response["username"].casefold() == username.casefold() and response["username"] != username:
        error = "wrong username, please make sure capitalization is correct."
        succes = False
    else:
        error = "wrong username/password."
        succes = False

    return succes, error

def copernicus_data_space_account(user_pw):
    """Check if the given or stored COPERNICUS_DATA_SPACE account is
    correct. Accounts can be created on https://dataspace.copernicus.eu

    Parameters
    ----------
    user_pw : tuple
        ("username", "password") to check.

    Returns
    -------
    tuple
        First item is `True` if the password works, otherwise `False`,
        seconds item is a error message.
    """

    username, password = user_pw

    try:

        data = {
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
            }
        
        r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                            data=data,
                            )
        r.raise_for_status()
        token = r.json()

        if "access_token" in token.keys():
            succes = True
            error = ""
        else:
            succes = False
            error = "Invalid username/password"

    except Exception as e:
        exception_args = getattr(e, "args", ["Invalid username/password"])
        error = exception_args[0]
        succes = False

    return succes, error

def cds_account(user_pw):
    """Check if the given or stored ECMWF key is 
    correct. Accounts can be created on https://cds.climate.copernicus.eu/#!/home.

    Parameters
    ----------
    user_pw : tuple
        ("username", "password") to check.

    Returns
    -------
    tuple
        First item is `True` if the password works, otherwise `False`,
        seconds item is a error message.
    """

    url, key = user_pw

    try:
        vrfy = {"NO": False, "YES": True}.get(os.environ.get("PYWAPOR_VERIFY_SSL", "YES"), True)
        c = cdsapi.Client(url = url, key = key, verify = vrfy, quiet = True)
        fp = os.path.join(pywapor.__path__[0], "test.zip")
        _ = c.retrieve(
            'sis-agrometeorological-indicators',
            {
                'format': 'zip',
                'variable': '2m_temperature',
                'statistic': '24_hour_mean',
                'year': '2005',
                'month': '11',
                'day': ['01'],
                'area': [2,-2,-2,2],
                'version': '1_1'
            },
            fp)
        try:
            os.remove(fp)
        except PermissionError:
            ... # windows...
        succes = True
        error = ""
    except Exception as e:
        exception_args = getattr(e, "args", ["wrong_token"])
        succes = False
        if len(exception_args) > 0:
            if "Client has not agreed to the required terms and conditions" in str(exception_args[0]):
                error = exception_args[0]
            else:
                error = str(e)
        else:
            error = "wrong key"

    return succes, error

def ecmwf_account(user_pw):
    """Check if the given or stored ECMWF key is 
    correct. Accounts can be created on https://cds.climate.copernicus.eu/#!/home.

    Parameters
    ----------
    user_pw : tuple
        ("username", "password") to check.

    Returns
    -------
    tuple
        First item is `True` if the password works, otherwise `False`,
        seconds item is a error message.
    """

    url, key = user_pw

    try:
        vrfy = {"NO": False, "YES": True}.get(os.environ.get("PYWAPOR_VERIFY_SSL", "YES"), True)
        c = cdsapi.Client(url = url, key = key, verify = vrfy, quiet = True)
        fp = os.path.join(pywapor.__path__[0], "test.zip")
        _ = c.retrieve(
            'sis-agrometeorological-indicators',
            {
                'format': 'zip',
                'variable': '2m_temperature',
                'statistic': '24_hour_mean',
                'year': '2005',
                'month': '11',
                'day': ['01'],
                'area': [2,-2,-2,2],
                'version': '1_1'
            },
            fp)
        try:
            os.remove(fp)
        except PermissionError:
            ... # windows...
        succes = True
        error = ""
    except Exception as e:
        exception_args = getattr(e, "args", ["wrong_token"])
        succes = False
        if len(exception_args) > 0:
            if "Client has not agreed to the required terms and conditions" in str(exception_args[0]):
                error = exception_args[0]
            else:
                error = str(e)
        else:
            error = "wrong key"

    return succes, error

if __name__ == "__main__":
    ...
    adjust_logger(True, r"/Users/hmcoerver/Desktop", "INFO")

    un_pw1= get("NASA")
    # print("NASA", un_pw1)
    # check1 = nasa_account(un_pw1)

    un_pw2 = get("TERRA")
    # print("TERRA", un_pw2)
    # check2 = terra_account(un_pw2)

    un_pw4 = get("ECMWF")
    # print("ECMWF", un_pw4)
    # check4 = ecmwf_account(un_pw4)

    un_pw6 = get("EARTHEXPLORER")
    # print("EARTHEXPLORER", un_pw6)
    # check6 = earthexplorer_account(un_pw6)

    un_pw7 = get("COPERNICUS_DATA_SPACE")
    # print("COPERNICUS_DATA_SPACE", un_pw7)
    # check7 = copernicus_data_space_account(un_pw7)

    un_pw8 = get("VIIRSL1")
    # print("VIIRSL1", un_pw8)
    # check8 = viirs_account(un_pw8)

    un_pw8 = get("LSASAF")
    # print("VIIRSL1", un_pw8)
    # check8 = viirs_account(un_pw8)

    un_pw9 = get("CDS")

    # print(check1, check2, check4, check6, check7, check8)