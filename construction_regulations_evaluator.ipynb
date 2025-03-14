!pip install langchain-openai
!pip install langchain-anthropic
!pip install langchain-google-genai
!pip install langchain-community
!pip install langchain-huggingface
!pip install rouge-score
!pip install pypdf
!pip install chromadb
!pip install unstructured

import json
import time
from datetime import datetime
import os
import logging
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredFileLoader
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from langchain.schema import Document

class Configuration:
    """Class to store system configuration. Initialized with default parameters"""
    def __init__(self):
        # Document fragmentation parameters
        self.chunk_size = 1000
        self.chunk_overlap = 200

        # Embeddings model
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

        # LLM model configuration
        self.temperature = 0.7
        self.max_tokens = 2000

        # Retrieval strategy
        self.chain_type = "stuff"  # Options: "stuff", "map_reduce", "refine", "map_rerank"
        self.search_type = "mmr"  # Options: "similarity", "mmr", "similarity_score_threshold"
        self.k = 4  # Number of fragments to retrieve

        # Model for similarity evaluation
        self.similarity_model_name = "all-mpnet-base-v2"


class DocumentProcessor:
    def __init__(self, text_path, configuration):
        logger.info(f"Initializing DocumentProcessor with file: {text_path}")
        self.text_path = text_path
        self.configuration = configuration
        self.embeddings = HuggingFaceEmbeddings(model_name=configuration.embedding_model_name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.setup_document_base()

    def setup_document_base(self):
        if not os.path.exists(self.text_path):
            raise FileNotFoundError(f"File not found: {self.text_path}")

         # Use UnstructuredFileLoader instead of reading file directly
        loader = UnstructuredFileLoader(self.text_path)
        # loader.load() already returns a list of Document objects
        documents = loader.load()

        logger.info(f"Document successfully loaded.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.configuration.chunk_size,
            chunk_overlap=self.configuration.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Document divided into {len(chunks)} chunks")

        self.vector_store = Chroma.from_documents(chunks, self.embeddings)
        logger.info("Vector document database successfully created")

    def as_retriever(self):
        """Gets the retriever with the specified configuration."""
        return self.vector_store.as_retriever(
            search_type=self.configuration.search_type,
            search_kwargs={"k": self.configuration.k}
        )


class ModelEvaluator:
    def __init__(self, document_base, configuration):
        logger.info("Initializing ModelEvaluator")
        self.document_base = document_base
        self.configuration = configuration
        self.results_file = "evaluation_results.json"
        self.similarity_model = SentenceTransformer(configuration.similarity_model_name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def evaluate_model(self, model_name, model_instance, test_questions, ground_truth):
        logger.info(f"Starting evaluation of model: {model_name}")

        results = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "precision": 0,
                "time_taken": 0,
                "quality_score": 0,
                "similarity_scores": [],
            },
            "answers": [],
            "configuration": {
                "chunk_size": self.configuration.chunk_size,
                "chunk_overlap": self.configuration.chunk_overlap,
                "embedding_model": self.configuration.embedding_model_name,
                "temperature": self.configuration.temperature,
                "chain_type": self.configuration.chain_type,
                "search_type": self.configuration.search_type,
                "k_docs": self.configuration.k
            }
        }

        qa_chain = RetrievalQA.from_chain_type(
            llm=model_instance,
            chain_type=self.configuration.chain_type,
            retriever=self.document_base.as_retriever(),
            return_source_documents=True
        )
        logger.info(f"QA Chain successfully created with type: {self.configuration.chain_type}")

        total_time = 0
        all_answers = []

        for i, (question, truth) in enumerate(zip(test_questions, ground_truth)):
            logger.info(f"Processing question {i+1}/{len(test_questions)}")

            start_time = time.time()
            answer = qa_chain({"query": question})
            end_time = time.time()

            time_taken = end_time - start_time
            total_time += time_taken

            similarity = self.calculate_similarity(answer["result"], truth)
            results["metrics"]["similarity_scores"].append(similarity)

            # Extract source document information for analysis
            source_docs_info = []
            for doc in answer.get("source_documents", []):
                source_docs_info.append({
                    "content_preview": doc.page_content[:100] + "...",
                    "metadata": doc.metadata
                })

            results["answers"].append({
                "question": question,
                "answer": answer["result"],
                "ground_truth": truth,
                "time_taken": time_taken,
                "similarity_score": similarity,
                "source_documents": source_docs_info
            })
            all_answers.append(answer)

            logger.info(f"Question {i+1} successfully processed")

        # Calculate final metrics
        results["metrics"]["time_taken"] = total_time / len(test_questions)
        average_cosine = np.mean([s["cosine_similarity"] for s in results["metrics"]["similarity_scores"]])
        average_rouge = np.mean([s["rouge_l"] for s in results["metrics"]["similarity_scores"]])
        results["metrics"]["quality_score"] = (average_cosine + average_rouge) / 2
        results["metrics"]["precision"] = np.mean([s["cosine_similarity"] for s in results["metrics"]["similarity_scores"]])

        self.save_results(results, model_name)
        return results

    def calculate_similarity(self, answer, truth):
        logger.info("Calculating similarity between answer and truth")

        answer_embedding = self.similarity_model.encode(answer)
        truth_embedding = self.similarity_model.encode(truth)
        cosine_similarity = np.inner(answer_embedding, truth_embedding) / (
            np.linalg.norm(answer_embedding) * np.linalg.norm(truth_embedding)
        )

        answer_words = answer.lower().split()
        truth_words = truth.lower().split()

        smoother = SmoothingFunction().method1
        bleu_score = sentence_bleu([truth_words], answer_words, smoothing_function=smoother)

        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(truth, answer)

        logger.info("Similarity successfully calculated")

        return {
            "cosine_similarity": float(cosine_similarity),
            "bleu_score": float(bleu_score),
            "rouge_l": float(scores["rougeL"].fmeasure),
        }

    def save_results(self, results, model_name):
        logger.info("Saving results")
        try:
            # Create a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{model_name}_{timestamp}.json"

            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
                logger.info(f"Results successfully saved in {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")


def run_evaluation(
    text_path,
    chunk_size,
    chunk_overlap,
    embedding_model,
    temperature,
    max_tokens,
    chain_type,
    search_type,
    k,
    similarity_model,
    models_to_use
):
    """Runs the evaluation with the specified parameters."""
    if models_to_use is None:
        models_to_use = ["gpt-4", "claude", "gemini"]

    logger.info("Starting main program with configuration:")
    logger.info(f"  Chunk size: {chunk_size}")
    logger.info(f"  Overlap: {chunk_overlap}")
    logger.info(f"  Embeddings model: {embedding_model}")
    logger.info(f"  Temperature: {temperature}")
    logger.info(f"  Chain type: {chain_type}")
    logger.info(f"  Search type: {search_type}")
    logger.info(f"  K documents: {k}")
    logger.info(f"  Models to evaluate: {', '.join(models_to_use)}")

    # Configure parameters
    configuration = Configuration()
    configuration.chunk_size = chunk_size
    configuration.chunk_overlap = chunk_overlap
    configuration.embedding_model_name = embedding_model
    configuration.temperature = temperature
    configuration.max_tokens = max_tokens
    configuration.chain_type = chain_type
    configuration.search_type = search_type
    configuration.k = k
    configuration.similarity_model_name = similarity_model

    # Check file existence
    if not os.path.exists(text_path):
        logger.error(f"File not found at: {text_path}")
        raise FileNotFoundError(f"File not found: {text_path}")

    test_questions = [
       "¿Cuáles son las categorías de superficie para las Tiendas de Productos Básicos según la clasificación?",
       "¿En qué categorías se clasifican los centros de Educación Superior según su capacidad de ocupantes?",
       "¿Cómo se clasifican las Torres, Antenas y Chimeneas?",
       "¿Cuales son las especificaciones para una recámara principal?",
       "¿Cuál es la superficie máxima permitida para los Centros Comerciales y Merca-dos de más de 5,000 m²?",
       "¿Cuántas camas o consultorios se permiten como máximo para los Hospitales?",
       "¿Cuál es el ancho mínimo requerido para el acceso principal en edificaciones de tipo Habitación?",

    ]

    ground_truth = [
       "Hasta 250 m² y más de 250 m²",
       "Hasta 250 ocupantes y más de 250 ocupantes",
       "Hasta 8 m de altura, de 8 m hasta 30 m de altura, y más de 30 m de altura",
       "La recámara principal requiere un área mínima de 7.00 m², con un lado mínimo de 2.50 m y una altura de 2.30 m",
       "Hasta 2500 m²",
       "Más de 10 camas o consultorios",
       "0.90 metros",

    ]

    try:
        processor = DocumentProcessor(text_path, configuration)
        logger.info("DocumentProcessor successfully initialized")

        evaluator = ModelEvaluator(processor, configuration)
        logger.info("ModelEvaluator successfully initialized")

        # Configure models according to selected ones
        models = {}

        if "gpt-4" in models_to_use:
            models["GPT-4"] = ChatOpenAI(
                model_name="gpt-4",
                temperature=configuration.temperature,
                max_tokens=configuration.max_tokens,
                api_key=os.getenv("OPENAI_API_KEY")
            )

        if "claude" in models_to_use:
            models["Claude"] = ChatAnthropic(
                model="claude-3-opus-20240229",
                temperature=configuration.temperature,
                max_tokens_to_sample=configuration.max_tokens,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        if "gemini" in models_to_use:
            models["Gemini"] = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=configuration.temperature,
                max_output_tokens=configuration.max_tokens,
                convert_system_message_to_human=True,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )

        if not models:
            logger.error("No valid model was selected")
            return

        logger.info(f"Models successfully initialized: {', '.join(models.keys())}")

        all_results = {}
        for model_name, model_instance in models.items():
            try:
                logger.info(f"Evaluating model: {model_name}")
                results = evaluator.evaluate_model(
                    model_name, model_instance, test_questions, ground_truth
                )
                all_results[model_name] = results

                print(f"\nResults for {model_name}:")
                print(json.dumps(results["metrics"], indent=2))
                logger.info(f"Evaluation completed for {model_name}")

                # Print individual answers, ground truth, and similarity metrics.
                print("\nIndividual answers:")
                for resp in results["answers"]:
                    print(f"\nQuestion: {resp['question']}")
                    print(f"Model's answer: {resp['answer']}")
                    print(f"Ground Truth: {resp['ground_truth']}")
                    print(f"Response time: {resp['time_taken']:.2f} seconds")
                    print(f"Cosine similarity: {resp['similarity_score']['cosine_similarity']:.4f}")
                    print(f"BLEU: {resp['similarity_score']['bleu_score']:.4f}")
                    print(f"ROUGE-L: {resp['similarity_score']['rouge_l']:.4f}")
                    print("-" * 30)

                    # Print source document fragments for analysis
                    print("Retrieved document fragments:")
                    for i, doc in enumerate(resp.get("source_documents", [])):
                        print(f"Fragment {i+1}: {doc['content_preview']}")
                    print("-" * 30)

            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {str(e)}")
                continue

        # Save comparative results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"comparative_results_{timestamp}.json", "w") as f:
            json.dump(all_results, f, indent=2)
            logger.info(f"Comparative results saved in comparative_results_{timestamp}.json")

        return all_results

    except Exception as e:
        logger.error(f"Error in run_evaluation: {str(e)}")
        raise


# Examples of function usage (uncomment as needed):

# Run with default parameters
# all_results = run_evaluation()

# Test with smaller chunks and greater overlap
# all_results = run_evaluation(chunk_size=500, chunk_overlap=250)

# Test with lower temperature for more deterministic answers
# all_results = run_evaluation(temperature=0.2)

# Test with a different chain strategy
# all_results = run_evaluation(chain_type="map_rerank", k=6)

# Test only with GPT-4 and Claude
# all_results = run_evaluation(models_to_use=["gpt-4", "claude"])

if __name__ == "__main__":
    # Custom configuration to run the program
    # Modify these values as needed
    run_evaluation(
        chunk_size=1500,         # Size of chunks
        chunk_overlap=200,      # Overlap between chunks
        temperature=0.7,        # Models temperature (lower = more deterministic)
        chain_type="stuff",     # Chain type: "stuff", "map_reduce", "refine", "map_rerank"
        search_type="mmr", # Search type: "similarity", "mmr"
        k=5,                    # Number of chunks to retrieve
        models_to_use=["gpt-4", "claude", "gemini"],  # Models to evaluate
        embedding_model="sentence-transformers/all-mpnet-base-v2",  # Embeddings model
        max_tokens=2000,        # Maximum tokens in responses
        similarity_model="all-mpnet-base-v2",  # Similarity model
        text_path='reglamento/codigompalags_libro_sexto.txt'
    )
