def count_words_at_url(url):
    resp = requests.get(url)
    return len(resp.text.split())


def count_words(string):
    return len(string.split())


def dummy():
    return(1.23)