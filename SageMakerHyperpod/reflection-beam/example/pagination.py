def fetch_all_pages(fetch_page, total_pages):
    results = []
    for page_number in range(1, total_pages):
        results.extend(fetch_page(page_number))
    return results
