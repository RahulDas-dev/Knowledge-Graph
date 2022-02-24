"""Modules Defines functional blocks for Downloading various text dataset."""

import concurrent.futures

import pandas as pd
from tqdm import tqdm
import wikipediaapi  # pip install wikipedia-api


def wikipedia_scrape(topic, links_count=10, verbose=True):
    """
    Download Pages related to given topic from Wikipedia .

        Parameters:
            topic(string): Topic Name to search for.
            links_count(int): Number of links to download.
            verbose(boolean): Display Progress bar while downloading.

        Returns:
            pandas dataframe with columns 
                topic: topic,
                content: Page Content,
                url: Source URL/Link,
                categories: As categories by wikipedia
    """
    
    wiki_api = wikipediaapi.Wikipedia(
        language="en", extract_format=wikipediaapi.ExtractFormat.WIKI
    ) 

    def donwload_reference_link(link):
        """Downloads individual links."""
        try:
            page = wiki_api.page(link)
            if page.exists():
                return {
                    "topic": link,
                    "content": page.text,
                    "url": page.fullurl,
                    "categories": list(page.categories.keys()),
                }
            else: 
                return None    
        except BaseException:
            return None

    #Extracting Main Page related to given topic.
    main_page = wiki_api.page(topic) 
    if  not main_page.exists():
        print(f"Wikipedia does not have a page on topic {topic}.")
        return None

    page_collection = [
        {
            "topic": topic,
            "content": main_page.text,
            "url": main_page.fullurl,
            "categories": list(main_page.categories.keys()),
        }
    ]  

    #listing down the reference links.
    page_links = list(main_page.links.keys()) 
    print(f'Refenece Links count {len(page_links)}') if verbose else None
    if  len(page_links) > links_count:
        page_links = page_links[:links_count]

    #setting up Progress-bar.    
    progress = (
        tqdm(desc="Collecting Data", unit="", total=len(page_links)) if verbose else None
    ) 
    
    #concurrently downloading pages from refrence link 
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor: 
        future_link = {executor.submit(donwload_reference_link, link): link for link in page_links}
        for future in concurrent.futures.as_completed(future_link):
            data = future.result()
            page_collection.append(data) if data else None
            progress.update(1) if verbose else None
    progress.close() if verbose else None
    dataframe = pd.DataFrame(page_collection)
    dataframe["categories"] = dataframe.categories.apply(lambda x: [item.replace("Category:","") for item in x])
    dataframe["topic"] = topic
    tqdm.pandas(desc="sentence counting:")
    dataframe['sentence_count'] = dataframe['content'].progress_apply(lambda x: len(x.split('.')))
    dataframe.sort_values(by=['sentence_count'],ascending=False,inplace=True,ignore_index=True)
    return dataframe