# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# import warnings
# warnings.filterwarnings("ignore", message="Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.")
# # Load environment variables from the .env file
# load_dotenv()
# api_key=os.getenv("GEMINI_API_KEY")
# # 1. Initialize the Gemini model (picks up GOOGLE_API_KEY from environment automatically)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.1-flash-lite",
#     temperature=0.7,
#     api_key=api_key
# )


# # 4. Invoke the test prompt
# response = llm.invoke("Latest AI News and Research Updates")

# print("AI Response:")
# print(response.content)


import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

logging.getLogger("google_genai.models").setLevel(logging.ERROR)

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7,
    google_api_key=api_key,
)

response = llm.invoke(
    "Explain the latest developments in AI research."
)

print("AI Response:")
print(response)
