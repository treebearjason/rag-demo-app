import argparse
from doc_handler import DocumentHander
import ollama

CHROMA_PATH = "chroma"

SYSTEM_PROMPT = '''
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
'''

USER_PROMPT = '''
Context: {}, Question: {}
'''

class QueryService():

    def __init__(self):
        self.doc_handler = DocumentHander()

 
    def main(self):
        # Create CLI.
        parser = argparse.ArgumentParser()
        parser.add_argument("query_text", type=str, help="The query text.")
        args = parser.parse_args()
        query_text = args.query_text
        self.query_rag(query_text)


    def search_docs(self, query_text: str):
        # Prepare the DB.
        db = self.doc_handler.get_chroma_db()

        # Search the DB.
        results = db.similarity_search_with_score(query_text, k=5)

        relevant_text = ""
        relevant_text_ids = []
        for doc, score in results:
            relevant_text += doc.page_content
            relevant_text_ids.append(doc.metadata.get("id", None))
        return relevant_text, relevant_text_ids

        # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        # prompt = prompt_template.format(context=context_text, question=query_text)
        # print(prompt)

        # model = Ollama(model="mistral")
        # response_text = model.invoke(prompt)

        # sources = [doc.metadata.get("id", None) for doc, _score in results]
        # formatted_response = f"Response: {response_text}\nSources: {sources}"
        # print(formatted_response)
        # return response_text
    
    def call_llm(self, context:str, prompt:str):
        response = ollama.chat(
        model="mistral",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": USER_PROMPT.format(context, prompt),
            },
        ],
    )
        for chunk in response:
            if chunk["done"] is False:
                yield chunk["message"]["content"]
            else:
                break
