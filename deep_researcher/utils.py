from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os
from pydantic import BaseModel
from typing import List, Any, Optional, Literal
import asyncio
import urllib.error
import urllib.parse
import urllib.request
import json
import time
from dotenv import load_dotenv
from deep_researcher.struct import (
    SearchResult,
    SearchResults)

load_dotenv()


def init_llm(
        provider: Literal["openai", "anthropic", "google", "ollama"],
        model: str,
        temperature: float = 1
):
    """
    Initialize and return a language model chat interface based on the specified provider.

    This function creates a chat interface for different LLM providers including OpenAI, 
    Anthropic, Google, and Ollama. It handles API key validation and configuration for
    each provider.

    Args:
        provider: The LLM provider to use. Must be one of "openai", "anthropic", "google", or "ollama".
        model: The specific model name/identifier to use with the chosen provider.
        temperature: Controls randomness in the model's output. Higher values (e.g. 0.8) make the output
                    more random, while lower values (e.g. 0.2) make it more deterministic. Defaults to 0.5.

    Returns:
        A configured chat interface for the specified provider and model.

    Raises:
        ValueError: If the required API key environment variable is not set for the chosen provider
                   (except for Ollama which runs locally).
    """
    if provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY is not set. Please set it in your environment variables.")
        return AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_DEPLOYMENT"],
            api_version=os.environ["AZURE_OPENAI_VERSION"],
            temperature=temperature,
        )
    elif provider == "anthropic":
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise ValueError("ANTHROPIC_API_KEY is not set. Please set it in your environment variables.")
        return ChatAnthropic(model=model, temperature=temperature, api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider == "google":
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY is not set. Please set it in your environment variables.")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=os.environ["GOOGLE_API_KEY"])
    elif provider == "ollama":
        return ChatOllama(model=model, temperature=temperature)


# noinspection PyBroadException
class PubtatorAPIWrapper(BaseModel):
    """
    Wrapper around PubMed+Pubtator API.
    based on https://api.python.langchain.com/en/latest/_modules/langchain_community/utilities/pubmed.html#PubMedAPIWrapper

    This wrapper will use the PubMed API to conduct searches and fetch
    document summaries. By default, it will return the document summaries
    of the top-k results of an input search.

    :param top_k_results: number of the top-scored document used for the PubMed tool
    :param max_retry: maximum number of retries for a request. Default is 5
    :param sleep_time: time to wait between retries
    :param email: email address to be used for the PubMed API

    """

    base_url_esearch: str = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    )
    base_url_pubtator: str = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"
    batch_size: int = 5
    max_retry: int = 5
    sleep_time: float = 0.2

    # Default values for the parameters
    top_k_results: int = 3
    email: str = "your_email@examples.com"

    def lazy_load(self, query: str) -> list[dict]:
        """
        Loading function for querying pubtator
        :param query:
        :return:
        """
        url = (
                self.base_url_esearch
                + "db=pubmed&term="
                + str({urllib.parse.quote(query)})
                + f"&retmode=json&retmax={max(self.top_k_results, 10)}&usehistory=y"
        )
        print(url)
        result = urllib.request.urlopen(url)
        text = result.read().decode("utf-8")
        json_text = json.loads(text)
        return self.retrieve_article(json_text["esearchresult"]["idlist"])

    def retrieve_article(self, uids: list, ) -> List[dict]:
        """
        Gets articles from Pubtator
        :param uids:
        :return:
        """
        print(uids)
        uids = uids[:20]
        if len(uids) == 0:
            return []
        params = {"pmids": ",".join(uids), "full": "true"}
        url = f"{self.base_url_pubtator}?{urllib.parse.urlencode(params)}"
        retry = 0
        while True:
            try:
                result = urllib.request.urlopen(url)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and retry < self.max_retry:
                    # Too Many Requests errors
                    # wait for an exponentially increasing amount of time
                    print(  # noqa: T201
                        f"Too Many Requests, "
                        f"waiting for {self.sleep_time:.2f} seconds..."
                    )
                    time.sleep(self.sleep_time)
                    self.sleep_time *= 2
                    retry += 1

                else:
                    print(url)
                    raise e

        text_dict = json.loads(result.read().decode())
        return self._parse_article(text_dict)

    @staticmethod
    def _parse_article(text_dict: dict) -> List[dict]:
        """
        Very messy due to structure of XML
        :param text_dict:
        :return:
        """
        clean_response = []
        for i in text_dict['PubTator3']:
            # initialise the fields that should be present
            paper_id = i["_id"].split('|')[0]
            title = ''
            abstract = ''
            content = []
            sections = []
            for index, q in enumerate(i['passages']):
                # We have section type which is the section and the type of which refers to the nature of the text
                section_type = q["infons"].get('section_type')
                if section_type is None:
                    if q["infons"]["type"] == "title":
                        title = q["text"]
                    elif q["infons"]["type"] == "abstract":
                        abstract = q["text"]
                    else:
                        pass
                # Let's store the main title and abstract separately as it may be more informative
                if (section_type == "TITLE") and (q["infons"]["type"] == "front"):
                    title = q["text"]
                if (section_type == "ABSTRACT") and (q["infons"]["type"] == "abstract"):
                    abstract = q["text"]
                if section_type in {'AUTH_CONT', 'COMP_INT', 'REF', 'SUPPL', 'FIG'}:
                    # These fields are skipped as they are unlikely to be informative
                    continue
                else:
                    # Data is stored as markdown as this may help with understanding
                    valid_infron = q["infons"]
                    if "title" in valid_infron['type']:
                        size_of_tuple = valid_infron['type'].split('_')
                        try:
                            if len(size_of_tuple) > 0:
                                size_of: int | Any = int(size_of_tuple[-1]) + 1
                            else:
                                size_of: int = 1
                        except Exception:
                            size_of = 1
                        hashed = ''.join(["#" for _ in range(int(size_of))])
                        content.append(f'{hashed} {q["text"]}')
                    else:
                        # For paragraphs store this also in a separate to possible help with context windows later
                        if (valid_infron['type'] == "paragraph") or (valid_infron['type'] == "abstract"):
                            sections.append(q["text"])
                        content.append(q["text"])
            v = '\n\n'.join(content)
            clean_response.append({"uid": paper_id,
                                   "Title": title,
                                   "Summary": abstract,
                                   "Full Text": v,
                                   "Sections": sections
                                   })
        return clean_response


async def pubtator_search_async(
        queries: List,
        top_k_results: int = 5,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        type_of: Optional[str] = None,
        doc_content_chars_max: int = 4000
) -> List[SearchResults]:
    """
    Performs PubTator (PubMed) searches concurrently and returns structured SearchResults.

    Args:
        queries (List): List of Query objects (with .query attribute)
        top_k_results (int): Max results per query
        email (str): Email address for NCBI
        api_key (str): PubMed API key
        type_of (str): "id_search" or None
        doc_content_chars_max (int): Max characters in returned content

    Returns:
        List[SearchResults]: One SearchResults per query
    """

    # Fallback to environment variable if no api_key passed
    api_key = api_key or os.getenv("PUBTATOR_API_KEY", "")

    async def process_single_query(query_obj):
        try:
            wrapper = PubtatorAPIWrapper(
                top_k_results=top_k_results,
                doc_content_chars_max=doc_content_chars_max,
                email=email or "your_email@example.com",
                api_key=api_key or ""
            )

            if type_of == 'id_search':
                docs = await asyncio.to_thread(wrapper.retrieve_article, query_obj.query)
            else:
                docs = await asyncio.to_thread(wrapper.lazy_load, query_obj.query)

            results_structured = []
            for doc in docs:
                uid = doc.get("uid")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""
                results_structured.append(
                    SearchResult(
                        url=url,
                        title=doc.get("Title", ""),
                        raw_content=doc.get("Full Text") or doc.get("Summary") or ""
                    )
                )

            return SearchResults(query=query_obj, results=results_structured)

        except Exception as e:
            print(f"Error processing PubTator query '{query_obj.query}': {e}")
            return SearchResults(query=query_obj, results=[])

    # Run all searches sequentially with a small delay to avoid rate limits
    search_results = []
    delay = 1.0
    for i, q in enumerate(queries):
        if i > 0:
            await asyncio.sleep(delay)
        res = await process_single_query(q)
        search_results.append(res)

        # Adjust delay adaptively
        delay = max(0.5, delay * 0.9) if res.results else min(5.0, delay * 1.5)

    return search_results
