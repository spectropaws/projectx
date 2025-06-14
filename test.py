#!/usr/bin/env python3
"""
Local RAG System using LlamaCPP with GPU acceleration
Requires the vector database created from the Google Colab notebook
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
import argparse
import json
from pathlib import Path

# Install required packages (run once)
# pip install llama-cpp-python[cuda] --upgrade --force-reinstall --no-cache-dir
# pip install sentence-transformers faiss-cpu

try:
    from llama_cpp import Llama
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install llama-cpp-python[cuda] sentence-transformers faiss-cpu")
    exit(1)

class RAGSystem:
    def __init__(self, 
                 model_path: str,
                 vector_db_path: str,
                 n_gpu_layers: int = -1,
                 n_ctx: int = 4096,
                 temperature: float = 0.7,
                 top_k: int = 5):
        """
        Initialize the RAG system with LlamaCPP and vector database
        
        Args:
            model_path: Path to the GGUF model file
            vector_db_path: Path to the vector database directory
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            n_ctx: Context window size
            temperature: Sampling temperature
            top_k: Number of top documents to retrieve
        """
        self.model_path = model_path
        self.vector_db_path = vector_db_path
        self.top_k = top_k
        
        print("Loading LlamaCPP model...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            temperature=temperature,
            verbose=False
        )
        
        print("Loading vector database...")
        self.load_vector_database()
        
        print("RAG system initialized successfully!")
        
    def load_vector_database(self):
        """Load the vector database created in Google Colab"""
        db_path = Path(self.vector_db_path)
        
        # Load FAISS index
        index_path = db_path / "vector_index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Vector index not found at {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        # Load texts
        texts_path = db_path / "texts.pkl"
        with open(texts_path, 'rb') as f:
            self.texts = pickle.load(f)
            
        # Load metadata
        metadata_path = db_path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
            
        # Load config
        config_path = db_path / "config.pkl"
        with open(config_path, 'rb') as f:
            self.config = pickle.load(f)
            
        # Initialize embedding model (same as used in Colab)
        model_name = self.config.get('model_name', 'all-MiniLM-L6-v2')
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        print(f"Loaded vector database with {len(self.texts)} documents")
        
    def retrieve_relevant_docs(self, query: str, k: int = None) -> List[Dict]:
        """
        Retrieve the most relevant documents for a query
        
        Args:
            query: The user query
            k: Number of documents to retrieve (defaults to self.top_k)
            
        Returns:
            List of dictionaries containing relevant documents and metadata
        """
        if k is None:
            k = self.top_k
            
        # Encode the query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid index
                results.append({
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'score': float(score),
                    'rank': i + 1
                })
                
        return results
    
    def create_context_prompt(self, query: str, relevant_docs: List[Dict]) -> str:
        """
        Create a prompt with context from retrieved documents
        """
        context_parts = []
        for doc in relevant_docs:
            context_parts.append(f"Document {doc['rank']}: {doc['text']}")
            
        context = "\n".join(context_parts)
        
        prompt = f"""<|start_header_id|>system<|end_header_id|>

You are a NixOS configuration expert. Generate ONLY valid NixOS configuration code based on the user's request. 

Rules:
1. Return ONLY the NixOS configuration block, nothing else
2. Start with {{ and end with }}
3. Do not include explanations, comments, or additional text
4. Use proper NixOS syntax and module structure
5. Include necessary imports if needed

Use the context information to provide accurate and relevant answers. If the context doesn't contain enough information to answer the question, or if the user prompt is not understandable, do not make assumptions, just output the following snippet:
{{
    # Write a comment here explaining that the user prompt is not understandable or the context doesn't provide the required information in one line 
}}

Context:
{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def generate_response(self, 
                         query: str, 
                         max_tokens: int = 512,
                         show_context: bool = False) -> Dict:
        """
        Generate a response using RAG
        
        Args:
            query: User query
            max_tokens: Maximum tokens to generate
            show_context: Whether to include retrieved context in response
            
        Returns:
            Dictionary containing response and metadata
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(query)
        
        if not relevant_docs:
            return {
                'response': "I couldn't find any relevant information to answer your question.",
                'relevant_docs': [],
                'context_used': False
            }
        
        # Create prompt with context
        prompt = self.create_context_prompt(query, relevant_docs)
        
        # Generate response
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["<|eot_id|>", "<|end_of_text|>"],
            echo=False
        )
        
        result = {
            'response': response['choices'][0]['text'].strip(),
            'relevant_docs': relevant_docs if show_context else [],
            'context_used': True,
            'prompt_tokens': response.get('usage', {}).get('prompt_tokens', 0),
            'completion_tokens': response.get('usage', {}).get('completion_tokens', 0)
        }
        
        return result
    
    def interactive_chat(self):
        """
        Start an interactive chat session
        """
        print("\n" + "="*60)
        print("RAG Chat System - Llama 3.2 3B with Vector Database")
        print("Type 'quit' to exit, 'help' for commands")
        print("="*60)
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    print("Goodbye!")
                    break
                    
                if query.lower() == 'help':
                    print("\nCommands:")
                    print("- quit/exit: Exit the chat")
                    print("- help: Show this help message")
                    print("- stats: Show system statistics")
                    continue
                    
                if query.lower() == 'stats':
                    print(f"\nSystem Statistics:")
                    print(f"Model: {self.model_path}")
                    print(f"Vector DB: {len(self.texts)} documents")
                    print(f"Embedding model: {self.config.get('model_name', 'all-MiniLM-L6-v2')}")
                    print(f"Top-K retrieval: {self.top_k}")
                    continue
                    
                if not query:
                    continue
                    
                print("\nThinking...")
                result = self.generate_response(query, show_context=True)
                
                print(f"\nAssistant: {result['response']}")
                
                # Show relevant documents
                if result['relevant_docs']:
                    print(f"\n📚 Retrieved {len(result['relevant_docs'])} relevant documents:")
                    for doc in result['relevant_docs'][:3]:  # Show top 3
                        print(f"  • Score: {doc['score']:.3f} | Line: {doc['metadata']['line_number']}")
                        print(f"    {doc['text'][:100]}...")
                        
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="RAG System with LlamaCPP")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--vector-db", required=True, help="Path to vector database directory")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="Number of GPU layers (-1 for all)")
    parser.add_argument("--context-size", type=int, default=4096, help="Context window size")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--query", help="Single query mode")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return
        
    if not os.path.exists(args.vector_db):
        print(f"Error: Vector database directory not found at {args.vector_db}")
        return
    
    # Initialize RAG system
    try:
        rag = RAGSystem(
            model_path=args.model,
            vector_db_path=args.vector_db,
            n_gpu_layers=args.gpu_layers,
            n_ctx=args.context_size,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        if args.query:
            # Single query mode
            result = rag.generate_response(args.query, show_context=True)
            print(f"Query: {args.query}")
            print(f"Response: {result['response']}")
            
            if result['relevant_docs']:
                print(f"\nRelevant documents:")
                for doc in result['relevant_docs']:
                    print(f"  • Score: {doc['score']:.3f} | Line: {doc['metadata']['line_number']}")
                    print(f"    {doc['text'][:150]}...")
        else:
            # Interactive mode
            rag.interactive_chat()
            
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have the correct GGUF model file")
        print("2. Verify the vector database was created correctly")
        print("3. Check GPU drivers and CUDA installation")
        print("4. Try reducing --gpu-layers if you get GPU errors")

if __name__ == "__main__":
    main()

# Example usage:
# python rag_system.py --model ./models/llama-3.2-3b-instruct.gguf --vector-db ./rag_vector_db
# python rag_system.py --model ./models/llama-3.2-3b-instruct.gguf --vector-db ./rag_vector_db --query "What is the main topic discussed in the document?"
