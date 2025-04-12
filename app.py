import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')


## Get the Groq API Key and url(YT or website) to be summarized
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

## Gemma Model Using Groq API
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def summarize_content():
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
        return
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video url or website url")
        return

    try:
        with st.spinner("Loading content..."):
            if "youtube.com" in generic_url:
                # Verify YouTube URL format
                if "youtube.com/watch?v=" not in generic_url and "youtu.be/" not in generic_url:
                    st.error("Invalid YouTube URL format. Please use a full YouTube video URL")
                    return
                
                try:
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=True,
                        continue_on_failure=False
                    )
                    docs = loader.load()
                    if not docs:
                        st.error("Failed to load YouTube video. Possible reasons:\n"
                                "1. Video is private/age-restricted\n"
                                "2. Invalid video ID\n"
                                "3. Network issues\n"
                                "Please check the URL and try again.")
                        return
                except Exception as e:
                    st.error(f"YouTube Error: {str(e)}\n"
                            "Note: Some videos cannot be loaded due to YouTube restrictions.")
                    return
            else:
                try:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    docs = loader.load()
                    if not docs:
                        st.error("Failed to load website content. Please check the URL and try again.")
                        return
                except Exception as e:
                    st.error(f"Error loading website content: {str(e)}")
                    return

        with st.spinner("Generating summary..."):
            try:
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                st.success(output_summary)
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if st.button("Summarize the Content from YT or Website"):
    summarize_content()
