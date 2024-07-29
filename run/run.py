


from datasets import load_dataset
import csv
import os
from griptape.drivers import PineconeVectorStoreDriver, OpenAiEmbeddingDriver, PongoRerankDriver, OpenAiChatPromptDriver
from griptape.config import StructureConfig
from griptape.structures import Agent
from griptape.rules import Rule
from dotenv import load_dotenv
from griptape.engines.rag import RagEngine
from griptape.engines.rag.stages import RetrievalRagStage, ResponseRagStage
from griptape.engines.rag.modules import VectorStoreRetrievalRagModule, TextChunksRerankRagModule, TextChunksResponseRagModule

load_dotenv()

embedding_driver = OpenAiEmbeddingDriver(model="text-embedding-3-small")

vector_store_driver = PineconeVectorStoreDriver(
    api_key=os.environ["PINECONE_API_KEY"],
    environment='us-west-2',
    index_name='hotpot-questions',
    embedding_driver=embedding_driver,
)

rerank_driver = PongoRerankDriver(api_key=os.environ["PONGO_API_KEY"], num_results=20)

prompt_driver = OpenAiChatPromptDriver(api_key=os.getenv('TOGETHER_API_KEY'), base_url='https://api.together.xyz/v1', model='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo')


answering_agent = Agent(
    config=StructureConfig(
        prompt_driver=prompt_driver,
    ),
    input="You are a question-answering system, you will be provided with a question and set of possibly relevant documents, then your task is to answer the question based only upon information in the documents.{{ args[0] }}\n\nInput question: {{ args[1] }}\n\n=== BEGIN INPUT DOCUMENTS ==={{ args[2] }}=== END INPUT DOCUMENTS ===",
    rules=[
        Rule("Provide answers as brief as possible."),
        Rule("If a question cannot be answered based on information in the input documents, say 'The question cannot be answered based on the provided documents.'"),
        Rule("Only use information in the provided input documents to answer the input question")
    ],
)

assessing_agent = Agent(
    config=StructureConfig(
        prompt_driver=prompt_driver,
    ),
    input='''You are an answer assessment system that can reply only with 'True' or 'False'. You will be provided with a question, ground truth, and proposed answer, then your task is to answer judge wether the proposed answer contains all of the information in the ground truth a it pertains to answering the input question.

Example question: Are Canada and the United states on the same continent?
Example ground truth: Yes
Example asnwer: Yes, both the United States and Canada are in North America.
Example output: True

Example question: What year is it?
Example ground truth: 1942
Example asnwer: 1998
Example output: False

Input question: {{ args[0] }}
Input grouth truth: {{ args[1] }}
Input proposed answer: {{ args[2] }}''',
    rules=[
        Rule("Only respond with 'True' or 'False'")
    ],
)

rag_engine = RagEngine(
    retrieval_stage=RetrievalRagStage(
        retrieval_modules=[
            VectorStoreRetrievalRagModule(
                vector_store_driver=vector_store_driver,
                query_params={
                    "namespace": "griptape",
                    "top_n": 200
                }
            )
        ],
        rerank_module=TextChunksRerankRagModule(rerank_driver=rerank_driver)
    ),
    response_stage=ResponseRagStage(
        response_module=TextChunksResponseRagModule()
        )
    )
    
q_to_documents = {}
dataset = load_dataset("hotpot_qa", 'distractor', split='validation')


for datapoint in dataset:
    doc_strings = []
    paragraph_index = 0
    for paragraph_list in datapoint['context']['sentences']:
        doc_string = ' '.join(paragraph_list)
        doc_strings.append(f"Title: {datapoint['context']['title'][paragraph_index]}\nBody: {doc_string.strip()}")
        paragraph_index += 1

    q_to_documents[datapoint['question']] = doc_strings





def write_result(filename, result_row):
    file_exists = os.path.isfile(f'./results/{filename}.csv')

    with open(f'./results/{filename}.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["question", "answer", "docs", "examples", "given_answer", "relevance_assessment"])
        writer.writerow(result_row)


def get_examples(example_type, question, answer):

    if example_type == "none":
        return "" 
    elif example_type == "static":
        return '''**Example 1**
Example question 1: What is the CEO of Pongo's favorite color?
Example documents 1:
===BEGIN EXAMPLE DOCUMENTS 1===
Source #1:
Title: Companies to Countries
Body: Pongo is based in the united states.

Source #2:
Title: CEOs to Companies
Body: Caleb is the CEO of Pongo

Source #3:
Title: CTOs to Companies
Body: Jamari is the CTO of Pongo
===END EXAMPLE DOCUMENTS 1===
Example answer 1: Orange

**Example 2**
Example question 2: Are Google and Microsoft headquartered in the same country?
Example documents 2:
===BEGIN EXAMPLE DOCUMENTS 2===
Source #1:
Title: Companies to Countries
Body: Pongo is based in the united states.

Source #2:
Title: CEOs to Companies
Body: Caleb is the CEO of Pongo

Source #3:
Title: CTOs to Companies
Body: Jamari is the CTO of Pongo

Source #4:
Title: Companies to countrie
Body: Google is based in the United states

Source #5:
Title: Companies to countrie
Body: Microsoft is based in the United states.
===END EXAMPLE DOCUMENTS 2===
Example answer 2: Yes, they are both based in the united states.
'''
    elif example_type == "dynamic":
        relevant_examples = rag_engine.process_query(f"Question: {question}\nAnswer: {answer}").output.to_dict()
        examples_string = ''
        # print(len(relevant_examples['value']))

        example_count = 1

        for example in relevant_examples['value']:
            if example_count > 3:
                return examples_string
            example_question = example['value'].split(": ", 1)[1].split("\n")[0]
            example_answer = example['value'].split("\n")[1].split(": ", 1)[1]

            if example_question == question:
                continue
            relevant_docs = q_to_documents[example_question]
            if len(''.join(relevant_docs)) > 10000: #skip abnormally large example sets to preserve context window
                print(f'Skipped "{example_question}" for being too long.')
                continue
            docs_string = ''

            doc_count = 0
            for doc in relevant_docs:
                doc_count +=1
                docs_string += f'Source #{doc_count}:\n{doc}\n\n'

            examples_string += f'''**Example {example_count}**
Example question {example_count}: Are Google and Microsoft headquartered in the same country?
Example documents {example_count}: {example_question}
===BEGIN EXAMPLE DOCUMENTS {example_count}===
{docs_string}
=== END EXAMPLE DOCUMENTS {example_count}===
Example answer {example_count}: {example_answer}\n\n'''
            example_count += 1
            # print(example_count)
            
        return examples_string
    else:
        print('invalid example_type provided')
        return ""




def do_run_dataset(filename = 'results', example_type = 'none', start_index = 0, end_index = 100):
    datapoint_index = start_index #long-running task so enable stopping and starting partway through

    

    while datapoint_index < len(dataset):
        if datapoint_index >= end_index:
            return

        if datapoint_index%10==0:
            print('datapoint index: '+str(datapoint_index))

        datapoint = dataset[datapoint_index]
        question = datapoint['question']
        answer = datapoint['answer']

        examples = get_examples(example_type, question, answer)

        documents_string = ''

        i = 0
        for doc in q_to_documents[question]:
            i+=1
            documents_string += f"Source #{i}:\n{doc}\n\n"

        given_answer = answering_agent.run(examples, question, documents_string).output


        assessment_score = assessing_agent.run(question, answer, given_answer).output

        write_result(filename=filename, result_row=[question, answer, documents_string, examples, given_answer, assessment_score])


        # except:
        #     print(f'had an error at: {datapoint_index}') 
        datapoint_index+=1
            





do_run_dataset('first-5k-dynamic-results.csv', example_type='static',end_index=5000)