from duckduckgo_search import DDGS

def search_duckduckgo(query, max_results=5):
    """
    Search DuckDuckGo for the given query and return a list of results (title, url, snippet).
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    'title': r.get('title', ''),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', '')
                })
    except Exception as e:
        results.append({'title': 'Error', 'url': '', 'snippet': str(e)})
    return results 