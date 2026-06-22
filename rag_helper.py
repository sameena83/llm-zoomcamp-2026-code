from openai import OpenAI

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        filename,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model="gpt-5.4-mini",
    ):
        self.index = index
        self.llm_client = llm_client
        self.filename = filename
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        boost_dict = {
            "filename": 3.0,
            "content": 0.5,
        }
        if self.filename:
            filter_dict = {"filename": self.filename}
        else:
            filter_dict = None

       

        return self.index.search(
            query=query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(f"Filename: {doc['filename']}")
            lines.append(f"Content: {doc['content']}")
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)

        return self.prompt_template.format(
            question=query,
            context=context,
        )

    def llm(self, prompt):
        response = self.llm_client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "developer",
                    "content": self.instructions,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        
        return response.output_text,response.usage
    
    

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer, usage = self.llm(prompt)
        return answer, usage