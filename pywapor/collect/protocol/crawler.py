import os
import urllib
from bs4 import BeautifulSoup
import warnings
import urllib.request
from joblib import Parallel, delayed
from functools import partial
import re
import numpy as np
import tqdm
from requests.exceptions import HTTPError
import time
import socket
import requests
from pywapor.general.logger import log
from cachetools import cached, TTLCache
import ssl

@cached(cache=TTLCache(maxsize=2048, ttl=3600))
def find_paths(url, regex, node_type = "a", tag = "href", filter = None, session = None):
    """Search for node an tag values on a webpage.

    Parameters
    ----------
    url : str
        URL of page to search.
    regex : str
        Regex to filter data.
    node_type : str, optional
        Node type to search for, by default "a".
    tag : str, optional
        Tage within `node` to search for, by default "href".
    filter : function, optional
        Function applied to each tag found, by default None.
    session : requests.Session, optional
        session to use to open `url`, by default None.

    Returns
    -------
    np.ndarray
        Array with all tags found.
    """
    if isinstance(session, type(None)):
        context = ssl._create_unverified_context()
        f = urllib.request.urlopen(url, context = context)
    else:
        file_object = session.get(url, stream = True)
        file_object.raise_for_status()
        f = file_object.content
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="It looks like you're parsing an XML document using an HTML parser.")
        soup = BeautifulSoup(f, features="xml")
    file_tags = soup.find_all(node_type, {tag: re.compile(regex)})
    if not isinstance(filter, type(None)):
        file_tags = np.unique([filter(tag) for tag in file_tags], axis = 0).tolist()
    return file_tags

def crawl(urls, regex, filter_regex, session, label_filter = None, list_out = False):
    """Search for tags on multiple pages.

    Parameters
    ----------
    urls : dict
        Values are urls to search.
    regex : str
        Regex to filter data.
    session : requests.Session
        session to use to open `url`.

    Returns
    -------
    dict
        URLs found.
    """

    def _filter(tag):
        url = tag["href"]
        date_str = re.findall(filter_regex, url)[0].replace("/", "")
        return (date_str, url)

    if list_out:
        crawled_urls = list()
    else:
        crawled_urls = dict()

    for url in tqdm.tqdm(urls.values(), delay = 20):
        raw_paths = find_paths(url, regex, filter = _filter, session = session)
        if list_out:
            crawled_urls += raw_paths
        else:
            crawled_urls = {**crawled_urls, **{dt: new_url for dt, new_url in raw_paths if dt in label_filter}}

    return crawled_urls

def _download_urls(urls, folder, session, fps = None, parallel = 0, headers = None,
                    max_tries = 10, wait_sec = 15):

    if isinstance(parallel, bool):
        parallel = {True: -1, False: 0}[parallel]

    if isinstance(fps, type(None)):
        fps = [os.path.join(folder, os.path.split(url)[-1]) for url in urls]
    else:
        assert len(fps) == len(urls)

    if parallel:
        backend = "loky"
        dler = partial(download_url, session = session, headers = headers, max_tries = max_tries, wait_sec = wait_sec)
        files = Parallel(n_jobs=parallel, backend = backend)(delayed(dler)(*x) for x in zip(urls, fps))
    else:
        files = list()
        for url, fp in zip(urls, fps):
            try:
                fp_out = download_url(url, fp, session = session, headers = headers, max_tries = max_tries, wait_sec = wait_sec)
                files.append(fp_out)
            except NameError as e:
                log.info(f"--> {e}")

    return files

def unzip(zipped):
    new, list_ = [], []
    for x in zipped:
        new.append(x[0])
        list_.append(x[1])
    return new, list_

def update_urls(urls, dled_files, relevant_fps, checker_function = None):
        
    if isinstance(urls, zip):
        urls, list_ = unzip(urls)
        fps_given = True
    else:
        list_ = [os.path.split(x)[-1] for x in urls]
        fps_given = False

    for fp in dled_files:

        if fp in relevant_fps:
            continue

        if isinstance(checker_function, type(None)):
            is_relevant = True
        else:
            is_relevant = checker_function(fp)

        if fps_given:
            x = fp
        else:
            x = os.path.split(fp)[-1]
        url_idx = list_.index(x) if x in list_ else None

        if os.path.isfile(fp):
            if not isinstance(url_idx, type(None)):
                _ = urls.pop(url_idx)
                _ = list_.pop(url_idx)
            if is_relevant:
                relevant_fps.append(fp)

    if fps_given:
        urls = zip(urls, list_)

    return urls, relevant_fps

def download_urls(urls, folder, session = None, fps = None, parallel = 0, 
                    headers = None, checker_function = None, 
                    max_tries = 10, wait_sec = 15, meta_max_tries = 2):
    """Download URLs to a folder.

    Parameters
    ----------
    urls : list
        URLs to download.
    folder : str
        Path to folder in which to download.
    session : request.Session, optional
        Session to use, by default None.
    fps : list, optional
        Filepaths to store each URL, should be same length as `urls`, ignores `folder`, by default None.
    parallel : int, optional
        Download URLs in parallel (currently not implemented), by default 0.
    headers : dict, optional
        Headers to include with requests, by default None.
    checker_function : function, optional
        Function to apply to downloaded files to see if they are valid, by default None.
    max_tries : int, optional
        Max number of tries to download a single file (in case of server side errors), by default 10.
    wait_sec : int, optional
        How many seconds to wait between retries, by default 15.
    meta_max_tries : int, optional
        Max number of tries to download all URLs before giving up, by default 2.

    Returns
    -------
    list
        Paths to downloaded files.
    """
    
    relevant_fps = list()
    try_n = 0
    try_again = True

    while try_again:

        dled_files = _download_urls(urls, folder, session, fps = fps, parallel = parallel, headers = headers, max_tries = max_tries, wait_sec = wait_sec)

        if not isinstance(fps, type(None)):
            urls = zip(urls, fps)

        urls, relevant_fps = update_urls(urls, dled_files, relevant_fps, checker_function = checker_function)

        if not isinstance(fps, type(None)):
            urls, fps = unzip(urls)

        try_again = len(urls) > 0 and try_n < meta_max_tries-1
        if try_again:
            try_n += 1
            log.info(f"--> Retrying to download {len(urls)} remaining urls.")

    if len(urls) > 0:
        log.warning(f"--> Didn't succeed to download {len(urls)} files.")
    return relevant_fps


def _download_url(url, fp, session = None, waitbar = True, headers = None):

    if os.path.isfile(fp):
        log.add().info(f"--> Reusing existing `{fp}` file.").sub()
        return fp

    folder, fn = os.path.split(fp)
    if not os.path.exists(folder):
        os.makedirs(folder)

    ext = os.path.splitext(fp)[-1]
    temp_fp = fp.replace(ext, "_temp")

    if isinstance(session, type(None)):
        session = requests.Session()

    vrfy = {"NO": False, "YES": True}.get(os.environ.get("PYWAPOR_VERIFY_SSL", "YES"), True)
    file_object = session.get(url, stream = True, headers = headers, verify = vrfy)
    file_object.raise_for_status()

    if "Content-Length" in file_object.headers.keys():
        tot_size = int(file_object.headers["Content-Length"])
    else:
        tot_size = None

    if waitbar:
        wb = tqdm.tqdm(unit='Bytes', unit_scale=True, leave = False, 
                        total = tot_size, desc = fn)

    with open(temp_fp, 'wb') as z:
        for data in file_object.iter_content(chunk_size=1024):
            size = z.write(data)
            if waitbar:
                wb.update(size)

    os.rename(temp_fp, fp)

    return fp

def download_url(url, fp, session = None, waitbar = True, headers = None, 
                    max_tries = 3, wait_sec = 15):
    """Download a URL to a file.

    Parameters
    ----------
    url : str
        URL to download.
    fp : str
        Path to file to download into.
    session : requests.Session, optional
        Session to use, by default None.
    waitbar : bool, optional
        Display a download progress bar, by default True.
    headers : dict, optional
        Headers to include with requests, by default None.
    max_tries : int, optional
        Max number of tries to download a single file (in case of server side errors), by default 10.
    wait_sec : int, optional
        How many seconds to wait between retries, by default 15.

    Returns
    -------
    str
        Path to downloaded file.

    Raises
    ------
    NameError
        Raised when `max_tries` is exceeeded.
    """
    try_n = 0
    succes = False
    while try_n <= max_tries-1 and not succes:
        if try_n > 0:
            waiter = int((try_n**1.2) * wait_sec)
            log.info(f"--> Trying to download {fp}, attempt {try_n+1} of {max_tries} in {waiter} seconds.")
            time.sleep(waiter)
        try:
            fp = _download_url(url, fp, session = session, waitbar = waitbar, headers = headers)
            succes = True
        except socket.timeout:
            log.info("--> Server connection timed out.")
        except HTTPError as e:
            error_body = getattr(getattr(e, "response", ""), "text", "")
            log.info(f"--> Server error {e} [{error_body}].")
        else:
            ...
        finally:
            try_n += 1
    
    if not succes:
        raise NameError(f"Could not download {url} after {max_tries} attempts.")

    return fp

if __name__ == "__main__":

    ...
